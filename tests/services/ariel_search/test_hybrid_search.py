"""Tests for the ARIEL qmd search module.

Every test here runs against a stubbed client, so the suite needs no sidecar and
no network. What the stub cannot fake — the wire protocol — is covered by the
client's own tests; what it exists for is the mapping this module owns: qmd's
ranking and filenames onto ARIEL's rows.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig, SearchModuleConfig
from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    QueryExpansion,
)
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

        self.caption_calls: list[dict[str, Any]] = []

    async def get_entries_by_ids(self, entry_ids: list[str]) -> list[dict[str, Any]]:
        self.requested.append(list(entry_ids))
        # Deliberately returned in table order, not request order: the real
        # repository's ``= ANY(...)`` makes no ordering promise either.
        return [self._by_id[eid] for eid in sorted(self._by_id) if eid in set(entry_ids)]

    async def caption_matches(
        self, entry_ids: list[str], model_id: str | None, **kwargs: Any
    ) -> dict[str, list[str]]:
        """No caption matches anything; records the call."""
        self.caption_calls.append({"entry_ids": list(entry_ids), "model_id": model_id, **kwargs})
        return {}


#: The English stop words the captions and queries below exercise.
_STOP_WORDS = frozenset({"a", "an", "and", "at", "in", "of", "on", "the", "to"})


def _lexemes(text: str) -> set[str]:
    """Approximate ``to_tsvector('english', text)`` lexemes for plain words."""
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOP_WORDS}


class CaptionRepository(StubRepository):
    """A stub whose attachments carry captions, matched by lexeme coverage.

    Mirrors the coverage form of ``ARIELRepository.caption_matches``: the
    caption's lexemes shared with the flattened query, capped at the original
    query's lexeme count, must reach ``ceil(min_fraction * n(original))``.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        captions: dict[str, dict[str, str]],
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__(entries)
        self._captions = captions
        self._error = error

    async def caption_matches(
        self, entry_ids: list[str], model_id: str | None, **kwargs: Any
    ) -> dict[str, list[str]]:
        await super().caption_matches(entry_ids, model_id, **kwargs)
        if self._error is not None:
            raise self._error
        original = _lexemes(kwargs["query_original"])
        flattened = _lexemes(kwargs.get("query_flattened") or kwargs["query_original"])
        needed = math.ceil(kwargs["min_fraction"] * len(original))
        out: dict[str, list[str]] = {}
        for entry_id in entry_ids:
            ids = sorted(
                attachment_id
                for attachment_id, caption in self._captions.get(entry_id, {}).items()
                if original and min(len(_lexemes(caption) & flattened), len(original)) >= needed
            )
            if ids:
                out[entry_id] = ids
        return out


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
    source_system: str = "Example eLog",
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

    def test_parameter_descriptors_report_the_configured_values(self):
        """The panel opens on what a query would do, not on what ships."""
        by_name = {
            p.name: p
            for p in get_parameter_descriptors(
                make_config({"rerank": False, "candidate_limit": 12})
            )
        }
        assert by_name["rerank"].default is False
        assert by_name["candidate_limit"].default == 12

    def test_malformed_config_falls_back_instead_of_raising(self):
        """Describing the module survives a key the query path would refuse."""
        by_name = {p.name: p for p in get_parameter_descriptors(make_config({"rerank": "junk"}))}
        assert by_name["rerank"].default is DEFAULT_RERANK
        assert by_name["candidate_limit"].default == DEFAULT_CANDIDATE_LIMIT

    def test_rerank_description_describes_the_cost_without_a_multiplier(self):
        """The panel hint explains the mechanism; perf ratios are not ours to promise."""
        rerank = {p.name: p for p in get_parameter_descriptors()}["rerank"]
        assert "4x" not in rerank.description
        assert "slower" in rerank.description.casefold()

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

    def test_shipped_defaults_are_rerank_on_and_forty_candidates(self):
        """Pinned here because the descriptors no longer pin them by construction."""
        assert DEFAULT_RERANK is True
        assert DEFAULT_CANDIDATE_LIMIT == 40
        assert HybridSearchSettings().rerank is True
        assert HybridSearchSettings().candidate_limit == 40

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
            "EX-2024",
            "entry with spaces",
            "übung",
            ".hidden",
            "%2F",
            "beam_current_setpoint",
            "EX-2003-0001",
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

        heading = render_entry(make_entry("EX-2003-0001")).splitlines()[0]

        assert heading == f"# {TITLE_ENTRY_PREFIX}EX-2003-0001"

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
            source_system="Example eLog",
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
            make_entry("1", source_system="Example eLog"),
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

    def __init__(
        self,
        probe_seconds: float = 0.2,
        query_seconds: float = 0.2,
        probe_release: threading.Event | None = None,
    ) -> None:
        self._probe = probe_seconds
        self._query = query_seconds
        self._probe_release = probe_release
        self.probe_released: bool | None = None
        self.threads: dict[str, str] = {}

    def is_available(self) -> bool:
        self.threads["is_available"] = threading.current_thread().name
        if self._probe_release is None:
            time.sleep(self._probe)
        else:
            # Blocks until released or the bound runs out; the bound is the
            # probe's duration, so a probe nobody releases still returns.
            self.probe_released = self._probe_release.wait(self._probe)
        return True

    def query(self, collection: str | None, text: str, **kwargs: Any) -> list[QMDSearchResult]:  # noqa: ARG002 - the QMD client query signature
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

        The probe blocks until another coroutine has made progress: a counter
        on the loop releases it after a few ticks. With the probe off the loop
        the counter runs and the probe returns released; with the probe back on
        the loop the counter can never tick, so the probe sits out its whole
        bound and returns unreleased. The verdict is which way the probe
        returned, never how many ticks fit in a wall-clock window, so a slow
        runner only delays the release rather than failing the test. Only the
        *probe* blocks — the query returns at once — so the release measures
        the probe alone.
        """
        released_after = 5
        release = threading.Event()

        async def counter() -> None:
            ticks = 0
            while True:
                await asyncio.sleep(0.01)
                ticks += 1
                if ticks >= released_after:
                    release.set()

        client = BlockingClient(probe_seconds=10.0, query_seconds=0.0, probe_release=release)
        ticker = asyncio.create_task(counter())
        try:
            await hybrid_search("beam", StubRepository([]), make_config(), client=client)
        finally:
            ticker.cancel()

        assert client.probe_released, (
            "event loop was starved during the health probe "
            f"(no {released_after} ticks within the probe's 10 s bound)"
        )


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
# Reranker fallback
# --------------------------------------------------------------------------


class RerankFaultClient(StubClient):
    """A client whose reranked queries fail and whose fast queries answer.

    Models the fault this fallback exists for: the sidecar is up — ``/health``
    answered at resolve time — but the reranker itself cannot serve the query,
    because its model is still loading after a restart or because the reranked
    query took the daemon down. The unreranked path still answers.
    """

    def __init__(
        self,
        hits: list[QMDSearchResult] | None = None,
        *,
        error: Exception | None = None,
        retry_error: Exception | None = None,
    ) -> None:
        super().__init__(hits)
        self._rerank_error = error or RuntimeError("reranker model is still loading")
        self._retry_error = retry_error

    def query(self, collection: str | None, text: str, **kwargs: Any) -> list[QMDSearchResult]:
        if kwargs.get("rerank"):
            self.calls.append({"collection": collection, "text": text, **kwargs})
            raise self._rerank_error
        if self._retry_error is not None:
            self.calls.append({"collection": collection, "text": text, **kwargs})
            raise self._retry_error
        return super().query(collection, text, **kwargs)


class TestRerankFallback:
    """A reranked-query failure degrades the ranking; it never breaks search."""

    @pytest.mark.asyncio
    async def test_failed_rerank_retries_without_the_reranker(self):
        client = RerankFaultClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        result = await hybrid_search("beam", repo, make_config({"rerank": True}), client=client)

        assert isinstance(result, ModuleOutput)
        assert [call["rerank"] for call in client.calls] == [True, False]
        ((entry, _, _),) = result.entries
        assert entry["entry_id"] == "1"

    @pytest.mark.asyncio
    async def test_the_retry_is_otherwise_the_same_query(self):
        """Only ``rerank`` changes — a narrower retry would answer differently."""
        client = RerankFaultClient([])

        await hybrid_search(
            "beam",
            StubRepository([]),
            make_config({"rerank": True}),
            client=client,
            max_results=7,
            candidate_limit=11,
        )

        first, retry = client.calls
        assert {key: value for key, value in first.items() if key != "rerank"} == {
            key: value for key, value in retry.items() if key != "rerank"
        }

    @pytest.mark.asyncio
    async def test_the_degraded_ranking_is_reported_as_a_warning(self):
        client = RerankFaultClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        result = await hybrid_search("beam", repo, make_config({"rerank": True}), client=client)

        (diagnostic,) = result.diagnostics
        assert diagnostic.level is DiagnosticLevel.WARNING
        assert diagnostic.source == "hybrid"
        assert "rerank" in diagnostic.message.lower()

    @pytest.mark.asyncio
    async def test_any_exception_triggers_the_retry(self):
        """The client raises whatever its transport raised; all of it falls back."""
        client = RerankFaultClient([make_hit("1")], error=TimeoutError("read timed out"))
        repo = StubRepository([make_entry("1")])

        result = await hybrid_search("beam", repo, make_config({"rerank": True}), client=client)

        assert isinstance(result, ModuleOutput)
        assert [call["rerank"] for call in client.calls] == [True, False]

    @pytest.mark.asyncio
    async def test_a_failed_retry_raises_the_retrys_error(self):
        """One retry, not a loop: if the fast path is down too, search is down."""
        client = RerankFaultClient(
            error=TimeoutError("read timed out"),
            retry_error=RuntimeError("qmd daemon is gone"),
        )

        with pytest.raises(RuntimeError, match="qmd daemon is gone"):
            await hybrid_search(
                "beam", StubRepository([]), make_config({"rerank": True}), client=client
            )
        assert [call["rerank"] for call in client.calls] == [True, False]

    @pytest.mark.asyncio
    async def test_a_working_reranker_queries_once_and_returns_a_bare_list(self):
        client = StubClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        results = await hybrid_search("beam", repo, make_config({"rerank": True}), client=client)

        assert isinstance(results, list)
        assert len(client.calls) == 1
        assert client.calls[0]["rerank"] is True

    @pytest.mark.asyncio
    async def test_an_unreranked_query_is_never_retried(self):
        """With the reranker off there is nothing to fall back to."""
        client = StubClient([make_hit("1")], error=RuntimeError("index missing"))

        with pytest.raises(RuntimeError, match="index missing"):
            await hybrid_search(
                "beam", StubRepository([]), make_config({"rerank": False}), client=client
            )
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_a_zero_hit_fallback_still_reports_the_warning(self):
        """The degraded ranking is news even when it ranked nothing."""
        client = RerankFaultClient([])

        result = await hybrid_search(
            "beam", StubRepository([]), make_config({"rerank": True}), client=client
        )

        assert isinstance(result, ModuleOutput)
        assert result.entries == []
        assert len(result.diagnostics) == 1

    @pytest.mark.asyncio
    async def test_one_output_carries_both_the_warning_and_the_expansion(self):
        client = RerankFaultClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        result = await hybrid_search(
            "ts fault",
            repo,
            make_config({"rerank": True}),
            client=client,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert result.expansion == AMBIGUOUS_GROUPS
        assert len(result.diagnostics) == 1


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


# --------------------------------------------------------------------------
# Vocabulary expansion
# --------------------------------------------------------------------------


AMBIGUOUS_GROUPS = (ExpansionGroup(original="ts", alternatives=("troubleshoot", "timing system")),)
AMBIGUOUS_EXPANSION = QueryExpansion(
    groups=AMBIGUOUS_GROUPS,
    flattened_text="ts fault troubleshoot timing system",
)


class TestVocabularyExpansion:
    """What the sidecar is sent, and what comes back, when expansion is active.

    Hybrid search matches on the whole query, so expansion is one substitution:
    the sidecar — and therefore its reranker — sees the flattened text instead
    of the raw query. Nothing is truncated on the way.
    """

    @pytest.mark.asyncio
    async def test_flattened_text_reaches_the_sidecar(self):
        client = StubClient([])

        await hybrid_search(
            "ts fault",
            StubRepository([]),
            make_config(),
            client=client,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert client.calls[0]["text"] == "ts fault troubleshoot timing system"

    @pytest.mark.asyncio
    async def test_raw_query_reaches_the_sidecar_without_expansion(self):
        """Without an expansion the call is byte-identical to today's."""
        client = StubClient([])

        await hybrid_search("ts fault", StubRepository([]), make_config(), client=client)

        assert client.calls[0]["text"] == "ts fault"

    @pytest.mark.asyncio
    async def test_long_query_is_not_truncated(self):
        """The 1000-char cap is keyword-only; hybrid queries go whole."""
        client = StubClient([])
        query = "beam loss " * 120  # 1200 characters

        await hybrid_search(query, StubRepository([]), make_config(), client=client)

        assert client.calls[0]["text"] == query

    @pytest.mark.asyncio
    async def test_without_expansion_a_bare_list_is_returned(self):
        """Direct callers keep the list they have always received."""
        client = StubClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        assert isinstance(results, list)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_with_expansion_a_module_output_carries_the_groups(self):
        client = StubClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        result = await hybrid_search(
            "ts fault",
            repo,
            make_config(),
            client=client,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.expansion == AMBIGUOUS_GROUPS
        assert result.diagnostics == ()
        ((entry, score, snippets),) = result.entries
        assert entry["entry_id"] == "1"
        assert isinstance(score, float)
        assert isinstance(snippets, list)

    @pytest.mark.asyncio
    async def test_zero_hit_path_keeps_the_shape(self):
        """A query the sidecar answers with nothing still answers in shape."""
        client = StubClient([])

        result = await hybrid_search(
            "ts fault",
            StubRepository([]),
            make_config(),
            client=client,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_blank_query_keeps_the_shape_without_reaching_the_sidecar(self):
        client = StubClient([])

        result = await hybrid_search(
            "   ",
            StubRepository([]),
            make_config(),
            client=client,
            query_expansion=AMBIGUOUS_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.entries == []
        assert client.calls == []

    def test_descriptor_opts_into_expansion(self):
        descriptor = get_tool_descriptor()

        assert descriptor.accepts_expansion is True

    def test_descriptor_declares_no_query_parser(self):
        """Hybrid search matches whole text — there is nothing to parse."""
        assert get_tool_descriptor().query_parser is None


# --------------------------------------------------------------------------
# Caption evidence
# --------------------------------------------------------------------------

#: An expansion of "orbit kick near BPM 7" that folds one alternative in.
KICK_EXPANSION = QueryExpansion(
    groups=(ExpansionGroup(original="kick", alternatives=("kicker",)),),
    flattened_text="orbit kick kicker near BPM 7",
)


class TestCaptionMatchedIds:
    """Hybrid marks the results whose attachment captions cover the query.

    The ids are evidence only: the sidecar's ordering and scores are kept, and a
    failing caption lookup degrades to no ids rather than a failed search.
    """

    @staticmethod
    def _repository(**kwargs: Any) -> CaptionRepository:
        return CaptionRepository(
            [make_entry("e1"), make_entry("e2")],
            {
                "e1": {"att-kick": "orbit kick at BPM 7", "att-other": "vacuum gauge readout"},
                "e2": {"att-rf": "RF cavity trip"},
            },
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_a_covering_caption_marks_its_attachment(self):
        """Image embedding is off (the default config): captions still match."""
        config = make_config()
        assert config.is_enhancement_module_enabled("image_embedding") is False
        repository = self._repository()

        results = await hybrid_search(
            "orbit kick near BPM 7",
            repository,
            config,
            client=StubClient([make_hit("e1", score=0.9), make_hit("e2", score=0.4)]),
        )

        assert isinstance(results, list)
        by_id = {entry["entry_id"]: entry for entry, _score, _snippets in results}
        assert by_id["e1"]["_matched_attachment_ids"] == ["att-kick"]
        assert "_matched_attachment_ids" not in by_id["e2"]

    @pytest.mark.asyncio
    async def test_the_call_carries_the_coverage_form(self):
        repository = self._repository()

        await hybrid_search(
            "orbit kick near BPM 7",
            repository,
            make_config(),
            client=StubClient([make_hit("e1"), make_hit("e2")]),
        )

        [call] = repository.caption_calls
        assert sorted(call["entry_ids"]) == ["e1", "e2"]
        assert call["model_id"] is None
        assert call["query_original"] == "orbit kick near BPM 7"
        assert call["query_flattened"] == "orbit kick near BPM 7"
        assert call["min_fraction"] == 0.5
        assert "tsquery_sql" not in call
        assert "pattern_bodies" not in call

    @pytest.mark.asyncio
    async def test_the_configured_caption_model_is_passed(self):
        """The module's enabled flag is ignored: existing captions stay searchable."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/ariel"},
                "search_modules": {"hybrid": {"enabled": True}},
                "enhancement_modules": {
                    "image_caption": {"enabled": False, "model": {"model_id": "cap-model"}}
                },
            }
        )
        repository = self._repository()

        await hybrid_search(
            "orbit kick near BPM 7",
            repository,
            config,
            client=StubClient([make_hit("e1")]),
        )

        assert repository.caption_calls[0]["model_id"] == "cap-model"

    @pytest.mark.asyncio
    async def test_with_vocabulary_expansion_the_flattened_text_is_counted(self):
        repository = self._repository()

        output = await hybrid_search(
            "orbit kick near BPM 7",
            repository,
            make_config(),
            client=StubClient([make_hit("e1"), make_hit("e2")]),
            query_expansion=KICK_EXPANSION,
        )

        assert isinstance(output, ModuleOutput)
        by_id = {entry["entry_id"]: entry for entry, _score, _snippets in output.entries}
        assert by_id["e1"]["_matched_attachment_ids"] == ["att-kick"]
        [call] = repository.caption_calls
        assert call["query_original"] == "orbit kick near BPM 7"
        assert call["query_flattened"] == "orbit kick kicker near BPM 7"

    @pytest.mark.asyncio
    async def test_ordering_and_scores_are_unchanged(self):
        hits = [make_hit("e2", score=0.8, snippet="1: rf"), make_hit("e1", score=0.3)]
        plain = await hybrid_search(
            "orbit kick near BPM 7",
            StubRepository([make_entry("e1"), make_entry("e2")]),
            make_config(),
            client=StubClient(hits),
        )
        marked = await hybrid_search(
            "orbit kick near BPM 7",
            self._repository(),
            make_config(),
            client=StubClient(hits),
        )

        assert isinstance(plain, list) and isinstance(marked, list)
        assert [(e["entry_id"], s, h) for e, s, h in marked] == [
            (e["entry_id"], s, h) for e, s, h in plain
        ]
        assert [e["entry_id"] for e, _s, _h in marked] == ["e2", "e1"]

    @pytest.mark.asyncio
    async def test_a_query_below_half_coverage_marks_nothing(self):
        repository = self._repository()

        results = await hybrid_search(
            "vacuum interlock trip sector four",
            repository,
            make_config(),
            client=StubClient([make_hit("e1"), make_hit("e2")]),
        )

        assert isinstance(results, list)
        assert all("_matched_attachment_ids" not in entry for entry, _s, _h in results)

    @pytest.mark.asyncio
    async def test_no_results_makes_no_caption_call(self):
        repository = self._repository()

        await hybrid_search(
            "orbit kick near BPM 7", repository, make_config(), client=StubClient([])
        )

        assert repository.caption_calls == []

    @pytest.mark.asyncio
    async def test_filtered_out_hits_are_not_looked_up(self):
        repository = CaptionRepository(
            [make_entry("e1", author="alice"), make_entry("e2", author="bob")],
            {"e1": {"att-kick": "orbit kick at BPM 7"}, "e2": {"att-b": "orbit kick at BPM 7"}},
        )

        await hybrid_search(
            "orbit kick near BPM 7",
            repository,
            make_config(),
            author="alice",
            client=StubClient([make_hit("e1"), make_hit("e2")]),
        )

        assert repository.caption_calls[0]["entry_ids"] == ["e1"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_kind", ["timeout", "database"])
    async def test_a_failing_lookup_warns_and_keeps_the_results(self, error_kind, caplog):
        """An unmigrated or slow store degrades to no ids, never to a failed search."""
        from osprey.services.ariel_search.exceptions import (
            DatabaseQueryError,
            SearchTimeoutError,
        )

        error = (
            SearchTimeoutError(
                "caption statement timed out", timeout_seconds=1.0, operation="caption_matches"
            )
            if error_kind == "timeout"
            else DatabaseQueryError('column "attachment_captions" does not exist')
        )
        repository = self._repository(error=error)

        with caplog.at_level("WARNING", logger="ariel"):
            results = await hybrid_search(
                "orbit kick near BPM 7",
                repository,
                make_config(),
                client=StubClient([make_hit("e1"), make_hit("e2")]),
            )

        assert isinstance(results, list)
        assert [entry["entry_id"] for entry, _s, _h in results] == ["e1", "e2"]
        assert all("_matched_attachment_ids" not in entry for entry, _s, _h in results)
        assert any("caption matching skipped" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_an_unmigrated_store_returns_results_without_ids(self):
        """A store without copy state answers ``{}``: results come back unmarked."""
        repository = StubRepository([make_entry("e1")])

        results = await hybrid_search(
            "orbit kick near BPM 7",
            repository,
            make_config(),
            client=StubClient([make_hit("e1")]),
        )

        assert isinstance(results, list)
        assert [entry["entry_id"] for entry, _s, _h in results] == ["e1"]
        assert "_matched_attachment_ids" not in results[0][0]


# --------------------------------------------------------------------------
# Picture lane integration
# --------------------------------------------------------------------------

from osprey.services.ariel_search.capabilities import attachments_capability  # noqa: E402
from osprey.services.ariel_search.search import fusion, image_lane  # noqa: E402
from osprey.services.ariel_search.search.fusion import ImageHit  # noqa: E402
from osprey.services.qmd import QMDClientError  # noqa: E402
from tests.services.ariel_search.conftest import _FakePool  # noqa: E402
from tests.services.ariel_search.llama_stub import MODEL as _LLAMA_MODEL  # noqa: E402


def lane_config(url: str = "http://127.0.0.1:1", *, enabled: bool = True) -> ARIELConfig:
    """A config with hybrid on and a llama-cpp ``image_embedding`` block at *url*."""
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost/ariel"},
            "search_modules": {"hybrid": {"enabled": True}},
            "enhancement_modules": {
                "image_embedding": {
                    "enabled": enabled,
                    "provider": {"name": "llama-cpp", "base_url": url},
                    "model": _LLAMA_MODEL,
                    "dimensions": 1024,
                }
            },
        }
    )
    return config


class LaneRepository(StubRepository):
    """A stub repository with an empty image table behind a fake pool."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        super().__init__(entries)
        self.pool = _FakePool(rows_for={"FROM ": []})


class FakeLane:
    """Stands in for ``image_lane.search_images``; records each call."""

    def __init__(self, result: dict[str, ImageHit] | None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, query, repository, config, *, fetch_limit):  # noqa: ARG002
        self.calls.append({"query": query, "fetch_limit": fetch_limit})
        return self.result


@pytest.fixture
def fake_lane(monkeypatch):
    def _install(result: dict[str, ImageHit] | None) -> FakeLane:
        lane = FakeLane(result)
        monkeypatch.setattr(image_lane, "search_images", lane)
        return lane

    return _install


@pytest.fixture
def fuse_spy(monkeypatch):
    calls: list[dict[str, Any]] = []
    real = fusion.fuse_lanes

    def _spy(text_hits, image_hits, **kwargs):
        calls.append({"text_hits": list(text_hits), "image_hits": dict(image_hits), **kwargs})
        return real(text_hits, image_hits, **kwargs)

    monkeypatch.setattr(fusion, "fuse_lanes", _spy)
    return calls


@pytest.fixture
def provider_spy(monkeypatch):
    calls: list[Any] = []

    def _resolve(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("the picture lane must not resolve a provider")

    monkeypatch.setattr(image_lane, "resolve_provider", _resolve)
    return calls


def _entries(result) -> list[tuple[dict[str, Any], float, list[str]]]:
    return list(result.entries) if isinstance(result, ModuleOutput) else list(result)


def _ids(result) -> list[str]:
    return [entry["entry_id"] for entry, _score, _snippets in _entries(result)]


def _strip_lane_keys(rows):
    """The rows with the keys only the picture lane adds removed."""
    out = []
    for entry, score, snippets in rows:
        clean = {k: v for k, v in entry.items() if k != "_matched_via"}
        out.append((clean, score, snippets))
    return out


def _picture_diagnostics(result) -> list[Any]:
    if not isinstance(result, ModuleOutput):
        return []
    return [d for d in result.diagnostics if d.message.startswith("Picture search unavailable")]


REQ4_TEXT = ["T1", "ORB", "T3"]
REQ4_IMAGES = {
    "ORB": ImageHit("att-orb", 0.70),
    "I2": ImageHit("att-i2", 0.68),
    "I3": ImageHit("att-i3", 0.66),
}


def _req4_fixture():
    client = StubClient(
        [make_hit(e, score=s) for e, s in zip(REQ4_TEXT, [0.9, 0.8, 0.7], strict=True)]
    )
    repo = LaneRepository([make_entry(e) for e in [*REQ4_TEXT, "I2", "I3"]])
    return client, repo


class TestPictureLaneFusion:
    """Requirement 4: the picture lane fused into the text ranking."""

    async def test_fused_order_matched_via_and_picture_ids(self, fake_lane, fuse_spy):
        lane = fake_lane(dict(REQ4_IMAGES))
        client, repo = _req4_fixture()

        result = await hybrid_search(
            "orbit plot", repo, lane_config(), client=client, max_results=10, include_images=True
        )

        rows = _entries(result)
        assert _ids(result) == ["ORB", "T1", "I2", "T3", "I3"]
        via = {entry["entry_id"]: entry["_matched_via"] for entry, _, _ in rows}
        assert via == {
            "ORB": ["image", "text"],
            "T1": ["text"],
            "I2": ["image"],
            "T3": ["text"],
            "I3": ["image"],
        }
        pictures = {entry["entry_id"]: entry.get("_matched_attachment_ids") for entry, _, _ in rows}
        assert pictures == {
            "ORB": ["att-orb"],
            "T1": None,
            "I2": ["att-i2"],
            "T3": None,
            "I3": ["att-i3"],
        }
        # Fused scores are normalised by the best: the top entry scores 1.0.
        assert rows[0][1] == pytest.approx(1.0)
        assert all(rows[i][1] >= rows[i + 1][1] for i in range(len(rows) - 1))
        # Image-only entries carry no snippet.
        assert {e["entry_id"]: s for e, _, s in rows}["I2"] == []
        # Both lanes hydrate in one read.
        assert len(repo.requested) == 1
        assert set(repo.requested[0]) == {"T1", "ORB", "T3", "I2", "I3"}
        assert fuse_spy[0]["cap"] == 4
        assert lane.calls == [{"query": "orbit plot", "fetch_limit": 10}]

    async def test_i3_ties_t3_and_is_ranked_after_it(self, fake_lane):
        fake_lane(dict(REQ4_IMAGES))
        client, repo = _req4_fixture()

        rows = _entries(
            await hybrid_search("q", repo, lane_config(), client=client, include_images=True)
        )

        scores = {entry["entry_id"]: score for entry, score, _ in rows}
        assert scores["I3"] == pytest.approx(scores["T3"])
        assert _ids(rows).index("T3") < _ids(rows).index("I3")

    async def test_picture_caption_ids_come_before_the_picture_id(self, fake_lane):
        fake_lane({"T1": ImageHit("att-pic", 0.9)})
        client = StubClient([make_hit("T1")])
        repo = CaptionRepository([make_entry("T1")], {"T1": {"att-cap": "orbit plot"}})
        repo.pool = _FakePool()

        rows = _entries(
            await hybrid_search(
                "orbit plot", repo, lane_config(), client=client, include_images=True
            )
        )

        assert rows[0][0]["_matched_attachment_ids"] == ["att-cap", "att-pic"]

    async def test_caption_and_picture_naming_the_same_id_list_it_once(self, fake_lane):
        fake_lane({"T1": ImageHit("att-cap", 0.9)})
        client = StubClient([make_hit("T1")])
        repo = CaptionRepository([make_entry("T1")], {"T1": {"att-cap": "orbit plot"}})

        rows = _entries(
            await hybrid_search(
                "orbit plot", repo, lane_config(), client=client, include_images=True
            )
        )

        assert rows[0][0]["_matched_attachment_ids"] == ["att-cap"]

    @pytest.mark.parametrize(("distance", "admitted"), [(0.30, True), (0.70, False)])
    async def test_similarity_floor_admits_070_and_drops_030(self, fake_lane, distance, admitted):
        fake_lane({"PIC": ImageHit("att-pic", 1 - distance)})
        client = StubClient([make_hit("T1")])
        repo = LaneRepository([make_entry("T1"), make_entry("PIC")])

        ids = _ids(
            await hybrid_search("q", repo, lane_config(), client=client, include_images=True)
        )

        assert ("PIC" in ids) is admitted
        assert "T1" in ids

    async def test_an_empty_lane_result_marks_text_hits_and_keeps_qmd_order(self, fake_lane):
        fake_lane({})
        hits = [make_hit("a", score=0.9), make_hit("b", score=0.4)]
        repo = LaneRepository([make_entry("a"), make_entry("b")])

        rows = _entries(
            await hybrid_search(
                "q", repo, lane_config(), client=StubClient(hits), include_images=True
            )
        )

        assert [(e["entry_id"], s, e["_matched_via"]) for e, s, _ in rows] == [
            ("a", 0.9, ["text"]),
            ("b", 0.4, ["text"]),
        ]

    async def test_no_text_hits_still_fuse_the_picture_matches(self, fake_lane):
        fake_lane({"PIC": ImageHit("att-pic", 0.8)})
        repo = LaneRepository([make_entry("PIC")])

        rows = _entries(
            await hybrid_search(
                "q", repo, lane_config(), client=StubClient([]), include_images=True
            )
        )

        assert [(e["entry_id"], e["_matched_via"]) for e, _, _ in rows] == [("PIC", ["image"])]

    async def test_image_only_entries_pass_the_filters(self, fake_lane):
        fake_lane({"MINE": ImageHit("a1", 0.8), "THEIRS": ImageHit("a2", 0.8)})
        repo = LaneRepository(
            [make_entry("T1"), make_entry("MINE"), make_entry("THEIRS", author="someone else")]
        )

        ids = _ids(
            await hybrid_search(
                "q",
                repo,
                lane_config(),
                client=StubClient([make_hit("T1")]),
                author="operator",
                include_images=True,
            )
        )

        assert "MINE" in ids
        assert "THEIRS" not in ids

    async def test_a_picture_hit_without_a_row_is_dropped(self, fake_lane):
        fake_lane({"GONE": ImageHit("a1", 0.8)})
        repo = LaneRepository([make_entry("T1")])

        ids = _ids(
            await hybrid_search(
                "q", repo, lane_config(), client=StubClient([make_hit("T1")]), include_images=True
            )
        )

        assert ids == ["T1"]

    async def test_the_lane_sees_the_typed_query_not_the_expansion(self, fake_lane):
        lane = fake_lane({})
        client = StubClient([make_hit("a")])
        expansion = QueryExpansion(
            groups=(ExpansionGroup(original="BPM", alternatives=("beam position monitor",)),),
            flattened_text="BPM beam position monitor",
        )

        await hybrid_search(
            "BPM",
            LaneRepository([make_entry("a")]),
            lane_config(),
            client=client,
            query_expansion=expansion,
            include_images=True,
        )

        assert lane.calls[0]["query"] == "BPM"
        assert client.calls[0]["text"] == "BPM beam position monitor"


class TestPictureLaneWindow:
    """Image-only entries never push fetched text hits out."""

    async def test_ten_text_hits_and_five_image_only_keep_all_ten_text_hits(self, fake_lane):
        images = {f"I{i}": ImageHit(f"a{i}", 0.70 - i * 0.001) for i in range(1, 6)}
        fake_lane(images)
        text = [f"T{i:02d}" for i in range(1, 11)]
        repo = LaneRepository([make_entry(e) for e in [*text, *images]])

        rows = _entries(
            await hybrid_search(
                "q",
                repo,
                lane_config(),
                client=StubClient([make_hit(e) for e in text]),
                max_results=10,
                include_images=True,
            )
        )

        text_rows = [e["entry_id"] for e, _, _ in rows if e["_matched_via"] != ["image"]]
        image_rows = [e["entry_id"] for e, _, _ in rows if e["_matched_via"] == ["image"]]
        assert sorted(text_rows) == text
        assert len(image_rows) == math.ceil(10 / 3)

    async def test_fusion_runs_over_every_filter_accepted_text_hit(self, fake_lane, fuse_spy):
        fake_lane({"I1": ImageHit("a1", 0.7)})
        text = [f"T{i:02d}" for i in range(1, 16)]
        repo = LaneRepository([make_entry(e) for e in [*text, "I1"]])

        rows = _entries(
            await hybrid_search(
                "q",
                repo,
                lane_config(),
                client=StubClient([make_hit(e) for e in text]),
                max_results=15,
                include_images=True,
            )
        )

        assert [eid for eid, _ in fuse_spy[0]["text_hits"]] == text
        assert len([e for e, _, _ in rows if e["_matched_via"] == ["text"]]) == 15

    async def test_at_most_max_results_text_lane_entries_are_kept(self, fake_lane):
        fake_lane({"T01": ImageHit("a1", 0.7)})
        text = [f"T{i:02d}" for i in range(1, 13)]
        repo = LaneRepository([make_entry(e) for e in text])

        # The fetch limit asks qmd for max_results hits; a client that answers
        # more stands for a window that over-fetched.
        client = StubClient([make_hit(e) for e in text])
        client.query = lambda collection, q, **kw: [make_hit(e) for e in text]  # type: ignore[method-assign]

        rows = _entries(
            await hybrid_search(
                "q", repo, lane_config(), client=client, max_results=10, include_images=True
            )
        )

        assert len(rows) == 10
        assert rows[0][0]["entry_id"] == "T01"


class TestPictureLaneOff:
    """With the lane off the B1 path runs unchanged."""

    async def test_explicit_false_runs_no_lane(self, monkeypatch, fuse_spy, provider_spy):
        lane_calls: list[Any] = []
        real = image_lane.search_images

        async def _spy(*args, **kwargs):
            lane_calls.append(args)
            return await real(*args, **kwargs)

        monkeypatch.setattr(image_lane, "search_images", _spy)
        hits = [make_hit("a", score=0.9), make_hit("b", score=0.5)]
        repo = LaneRepository([make_entry("a"), make_entry("b")])

        result = await hybrid_search(
            "q", repo, lane_config(), client=StubClient(hits), include_images=False
        )

        assert lane_calls == []
        assert provider_spy == []
        assert fuse_spy == []
        assert isinstance(result, list)
        assert all("_matched_via" not in e for e, _, _ in result)
        assert [(e["entry_id"], s) for e, s, _ in result] == [("a", 0.9), ("b", 0.5)]

    async def test_disabled_module_and_unset_flag_runs_no_lane(self, fake_lane, fuse_spy):
        lane = fake_lane({"x": ImageHit("a", 0.9)})

        result = await hybrid_search(
            "q",
            LaneRepository([make_entry("a")]),
            lane_config(enabled=False),
            client=StubClient([make_hit("a")]),
        )

        assert lane.calls == []
        assert fuse_spy == []
        assert all("_matched_via" not in e for e, _, _ in result)

    async def test_unset_flag_with_the_module_enabled_runs_the_lane(self, fake_lane):
        lane = fake_lane({})

        await hybrid_search(
            "q",
            LaneRepository([make_entry("a")]),
            lane_config(),
            client=StubClient([make_hit("a")]),
        )

        assert len(lane.calls) == 1

    async def test_lane_off_keeps_the_early_stop_at_max_results(self):
        hits = [make_hit(e) for e in ["a", "b", "c"]]
        repo = LaneRepository([make_entry(e) for e in ["a", "b", "c"]])

        result = await hybrid_search(
            "q", repo, lane_config(), client=StubClient(hits), max_results=2, include_images=False
        )

        assert _ids(result) == ["a", "b"]


class TestPictureLaneFailure:
    """A failed or cooling-down lane leaves qmd's output plus one diagnostic."""

    async def test_failed_lane_gives_the_text_result_plus_the_diagnostic(self, fake_lane, fuse_spy):
        hits = [make_hit("a", score=0.9), make_hit("b", score=0.5), make_hit("c", score=0.1)]
        rows = [make_entry("a"), make_entry("b"), make_entry("c")]

        off = await hybrid_search(
            "q",
            LaneRepository(rows),
            lane_config(),
            client=StubClient(hits),
            max_results=2,
            include_images=False,
        )
        fake_lane(None)
        failed = await hybrid_search(
            "q",
            LaneRepository([make_entry(r["entry_id"]) for r in rows]),
            lane_config(),
            client=StubClient(hits),
            max_results=2,
            include_images=True,
        )

        assert isinstance(failed, ModuleOutput)
        assert list(failed.entries) == list(off)
        assert all("_matched_via" not in e for e, _, _ in failed.entries)
        assert fuse_spy == []
        (diag,) = failed.diagnostics
        assert diag.level is DiagnosticLevel.WARNING
        assert diag.message.startswith("Picture search unavailable")
        assert diag.category == "picture_search"

    async def test_failed_lane_with_a_rerank_fallback_carries_both_diagnostics(self, fake_lane):
        fake_lane(None)

        class FlakyReranker(StubClient):
            def query(self, collection, text, **kwargs):
                if kwargs.get("rerank"):
                    raise RuntimeError("reranker down")
                return super().query(collection, text, **kwargs)

        result = await hybrid_search(
            "q",
            LaneRepository([make_entry("a")]),
            lane_config(),
            client=FlakyReranker([make_hit("a")]),
            rerank=True,
            include_images=True,
        )

        assert [d.category for d in result.diagnostics] == ["rerank", "picture_search"]

    async def test_qmd_client_error_leaves_no_pending_lane_task(self):
        held: list[int] = []
        tasks: list[asyncio.Task[Any]] = []

        async def _slow_lane(*_args: Any, **_kwargs: Any) -> dict[str, ImageHit]:
            tasks.append(asyncio.current_task())
            held.append(1)  # a pooled connection checked out
            try:
                await asyncio.sleep(30)
            finally:
                held.pop()
            return {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(image_lane, "search_images", _slow_lane)
            client = StubClient([make_hit("a")], error=QMDClientError("sidecar broke"))
            with pytest.raises(QMDClientError):
                await hybrid_search(
                    "q",
                    LaneRepository([make_entry("a")]),
                    lane_config(),
                    client=client,
                    rerank=False,
                    include_images=True,
                )

        assert len(tasks) == 1
        assert tasks[0].done()
        assert tasks[0].cancelled()
        assert held == []
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert tasks[0] not in pending

    async def test_qmd_down_with_picture_hits_returns_them_with_a_diagnostic(self, fake_lane):
        fake_lane({"PIC": ImageHit("att-pic", 0.8)})
        client = StubClient(available=False)

        result = await hybrid_search(
            "q",
            LaneRepository([make_entry("PIC")]),
            lane_config(),
            client=client,
            include_images=True,
        )

        assert isinstance(result, ModuleOutput)
        assert [(e["entry_id"], e["_matched_via"]) for e, _, _ in result.entries] == [
            ("PIC", ["image"])
        ]
        assert result.entries[0][0]["_matched_attachment_ids"] == ["att-pic"]
        (diag,) = result.diagnostics
        assert diag.message.startswith("Text ranking unavailable — picture matches only")
        assert client.calls == []

    @pytest.mark.parametrize("lane_result", [{}, None])
    async def test_qmd_down_without_picture_hits_still_raises(self, fake_lane, lane_result):
        fake_lane(lane_result)

        with pytest.raises(QMDUnavailableError):
            await hybrid_search(
                "q",
                LaneRepository([]),
                lane_config(),
                client=StubClient(available=False),
                include_images=True,
            )

    async def test_qmd_down_with_the_lane_off_raises_as_before(self, fake_lane):
        lane = fake_lane({"PIC": ImageHit("a", 0.9)})

        with pytest.raises(QMDUnavailableError):
            await hybrid_search(
                "q",
                LaneRepository([make_entry("PIC")]),
                lane_config(),
                client=StubClient(available=False),
                include_images=False,
            )
        assert lane.calls == []


@pytest.fixture
def lane_clock(monkeypatch):
    class _Clock:
        now = 1000.0

        def __call__(self) -> float:
            return self.now

    clock = _Clock()
    monkeypatch.setattr(image_lane, "_now", clock)
    return clock


def _closed_port_url() -> str:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


class TestPictureLaneAgainstAServer:
    """Owner ruling 17: no llama-server, or the wrong one, answers text-only."""

    async def test_closed_port_answers_text_only_fast_and_reports_unreachable(
        self, llama_stub, lane_clock
    ):
        # llama_stub empties LLAMA_CPP_HOST and the container fallbacks, so the
        # bound does not depend on the host's name resolution.
        config = lane_config(_closed_port_url())
        hits = [make_hit("a", score=0.9), make_hit("b", score=0.5)]

        def repo() -> LaneRepository:
            return LaneRepository([make_entry("a"), make_entry("b")])

        started = time.monotonic()
        off = await hybrid_search(
            "q", repo(), config, client=StubClient(hits), include_images=False
        )
        off_s = time.monotonic() - started

        started = time.monotonic()
        first = await hybrid_search("q", repo(), config, client=StubClient(hits))
        first_s = time.monotonic() - started

        assert first_s - off_s < 1.0
        assert isinstance(first, ModuleOutput)
        assert list(first.entries) == list(off)
        assert len(_picture_diagnostics(first)) == 1
        caps = attachments_capability(config)
        assert caps["picture_search"] is True
        assert caps["picture_search_unavailable"] == "unreachable"

        lane_clock.now += 0.5
        started = time.monotonic()
        second = await hybrid_search("q", repo(), config, client=StubClient(hits))
        assert time.monotonic() - started < 1.0
        assert list(second.entries) == list(off)
        assert len(_picture_diagnostics(second)) == 1

        # The cooldown ends, but no query has tried the lane again.
        lane_clock.now += image_lane.IMAGE_LANE_COOLDOWN_S + 1
        assert attachments_capability(config)["picture_search_unavailable"] == "unreachable"

        stub = llama_stub()
        working = lane_config(stub.url)
        result = await hybrid_search("q", repo(), working, client=StubClient(hits))
        assert _picture_diagnostics(result) == []
        assert [e["_matched_via"] for e, _, _ in _entries(result)] == [["text"], ["text"]]
        assert attachments_capability(working)["picture_search_unavailable"] is None

    async def test_a_server_serving_another_model_answers_text_only(self, llama_stub):
        stub = llama_stub()
        stub.alias = "other"
        config = lane_config(stub.url)

        result = await hybrid_search(
            "q", LaneRepository([make_entry("a")]), config, client=StubClient([make_hit("a")])
        )

        assert _ids(result) == ["a"]
        assert len(_picture_diagnostics(result)) == 1
        assert all("_matched_via" not in e for e, _, _ in _entries(result))
        assert attachments_capability(config)["picture_search_unavailable"] == "model"
        assert stub.embeddings == []
