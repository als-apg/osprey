"""Tests for the ARIEL ``jev`` search module.

The lexical stage is stubbed and the endpoint is stubbed: what is under test is
the part between them — how a batched set of typed answers becomes an ordering,
and what the operator sees when the endpoint does not answer at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.services.ariel_search.config import SearchModuleConfig
from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search import jev as jev_module
from osprey.services.ariel_search.search.base import ModuleOutput
from osprey.services.ariel_search.search.jev import (
    MAX_CANDIDATE_LIMIT,
    RELEVANCE_LEVELS,
    JevSearchSettings,
    blend_scores,
    build_questions,
    build_state,
    jev_search,
    normalized_score,
    recency_priors,
)
from osprey.services.jev import JevAnswer, JevError, JevResult


def _config(**settings: Any) -> Any:
    """A config carrying only what this module reads."""
    return SimpleNamespace(
        search_modules={"jev": SearchModuleConfig(enabled=True, settings=dict(settings))}
    )


def _entry(entry_id: str, text: str, *, day: int = 1) -> dict[str, Any]:
    """One logbook entry, with only the fields the state carries."""
    return {
        "entry_id": entry_id,
        "author": "operator",
        "timestamp": datetime(2026, 3, day, 12, 0, tzinfo=UTC),
        "raw_text": text,
        "summary": "",
        "keywords": [],
        "metadata": {},
    }


def _rows(count: int) -> list[tuple[dict[str, Any], float, list[str]]]:
    """A lexical result set, best-ranked first."""
    return [
        (_entry(f"e{index}", f"entry {index}", day=index + 1), 1.0 - index / 10, [])
        for index in range(count)
    ]


class _StubClient:
    """A JevClient stand-in that replays one prepared reply."""

    def __init__(self, result: Any = None, *, available: bool = True):
        self._result = result
        self._available = available
        self.state: Any = None
        self.questions: Any = None

    def is_available(self) -> bool:
        return self._available

    async def ask(self, state: Any, questions: Any) -> JevResult:
        self.state = state
        self.questions = questions
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _score(level: float, *, levels: int = len(RELEVANCE_LEVELS)) -> JevAnswer:
    """A score answer on the module's own rubric, 1-indexed as documented."""
    return JevAnswer(
        type="score",
        value=level,
        confidence=0.9,
        legend={str(index + 1): "" for index in range(levels)},
    )


def _result(answers: dict[str, JevAnswer]) -> JevResult:
    """A decoded reply with plausible usage figures."""
    return JevResult(
        answers=answers, model="jev-latest", input_tokens=7800, output_tokens=90, latency_ms=151
    )


@pytest.fixture
def stub_lexical(monkeypatch: pytest.MonkeyPatch):
    """Replace the lexical stage with a recorder returning a fixed pool."""
    calls: list[dict[str, Any]] = []

    def _install(rows: list[Any]):
        async def _keyword_search(query, repository, config, **kwargs):
            calls.append({"query": query, **kwargs})
            return ModuleOutput(entries=list(rows))

        monkeypatch.setattr(jev_module, "keyword_search", _keyword_search)
        return calls

    return _install


class TestSettings:
    """The settings block is read strictly, like every other module's."""

    def test_an_absent_block_yields_the_shipped_defaults(self):
        settings = JevSearchSettings.from_ariel_config(None)
        assert settings.candidate_limit == jev_module.DEFAULT_CANDIDATE_LIMIT
        assert settings.classify is True

    def test_a_present_key_is_honoured(self):
        settings = JevSearchSettings.from_ariel_config(_config(candidate_limit=12, classify=False))
        assert settings.candidate_limit == 12
        assert settings.classify is False

    def test_a_number_written_as_a_string_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"search_modules\.jev\.settings\.candidate_limit"):
            JevSearchSettings.from_ariel_config(_config(candidate_limit="30"))

    def test_a_boolean_written_as_a_string_is_refused(self):
        with pytest.raises(ValueError, match=r"settings\.classify must be a boolean"):
            JevSearchSettings.from_ariel_config(_config(classify="false"))

    def test_an_out_of_range_weight_is_refused(self):
        with pytest.raises(ValueError, match=r"lexical_weight must be a number in \[0, 1\]"):
            JevSearchSettings.from_ariel_config(_config(lexical_weight=1.5))

    def test_a_pool_beyond_the_ceiling_is_refused(self):
        with pytest.raises(ValueError, match="candidate_limit"):
            JevSearchSettings.from_ariel_config(_config(candidate_limit=MAX_CANDIDATE_LIMIT + 1))


class TestRequestShape:
    """One state, one question per candidate, asked in a single request."""

    def test_the_state_carries_the_query_and_an_indexed_pool(self):
        state = build_state("beam loss", [_entry("a", "x" * 900)], snippet_chars=100)
        assert state["query"] == "beam loss"
        assert state["candidates"][0]["index"] == 0
        assert state["candidates"][0]["entry_id"] == "a"

    def test_entry_text_is_truncated_to_the_configured_budget(self):
        state = build_state("q", [_entry("a", "x" * 900)], snippet_chars=100)
        assert len(state["candidates"][0]["text"]) == 100

    def test_one_relevance_question_per_candidate_plus_the_classifications(self):
        questions = build_questions(3, classify=True)
        assert sum(1 for key in questions if key.startswith("cand_")) == 3
        assert {"query_shape", "query_order"} <= set(questions)

    def test_classification_can_be_switched_off(self):
        questions = build_questions(3, classify=False)
        assert set(questions) == {"cand_0", "cand_1", "cand_2"}


class TestScoreNormalization:
    """A level means nothing until it is placed in its own range."""

    def test_the_legend_decides_the_range(self):
        assert normalized_score(_score(5), len(RELEVANCE_LEVELS)) == 1.0
        assert normalized_score(_score(1), len(RELEVANCE_LEVELS)) == 0.0
        assert normalized_score(_score(3), len(RELEVANCE_LEVELS)) == 0.5

    def test_a_zero_indexed_legend_is_read_as_such(self):
        answer = JevAnswer(type="score", value=0, legend={"0": "", "4": ""})
        assert normalized_score(answer, 5) == 0.0

    def test_an_unreadable_score_scores_nothing_rather_than_guessing(self):
        answer = JevAnswer(type="score", value=None, legend={})
        assert normalized_score(answer, 5) == 0.0


class TestBlending:
    """How a relevance becomes an ordering."""

    def test_relevance_dominates_but_the_lexical_rank_breaks_ties(self):
        # Two candidates the model rates identically keep the lexical order.
        blended = blend_scores([0.5, 0.5], lexical_weight=0.2, recency_weight=0.0)
        assert blended[0] > blended[1]

    def test_a_confident_model_can_overturn_the_lexical_rank(self):
        blended = blend_scores([0.0, 1.0], lexical_weight=0.2, recency_weight=0.0)
        assert blended[1] > blended[0]

    def test_a_heavy_lexical_weight_protects_the_top_row(self):
        blended = blend_scores([0.0, 1.0], lexical_weight=0.9, recency_weight=0.0)
        assert blended[0] > blended[1]

    def test_recency_lifts_the_newest_candidate(self):
        without = blend_scores([0.5, 0.5], lexical_weight=0.2, recency_weight=0.0)
        with_recency = blend_scores(
            [0.5, 0.5], lexical_weight=0.2, recency_weight=0.5, recency_ranks=[0.0, 1.0]
        )
        assert without[1] < without[0]
        assert with_recency[1] > with_recency[0]

    def test_an_empty_pool_blends_to_nothing(self):
        assert blend_scores([], lexical_weight=0.2, recency_weight=0.0) == []

    def test_recency_priors_rank_newest_first_regardless_of_pool_order(self):
        pool = [_entry("old", "", day=1), _entry("new", "", day=9), _entry("mid", "", day=5)]
        assert recency_priors(pool) == [0.0, 1.0, 0.5]


class TestSearch:
    """End to end over the module, with both stages stubbed."""

    @pytest.mark.asyncio
    async def test_the_model_reorders_the_lexical_pool(self, stub_lexical):
        rows = _rows(3)
        stub_lexical(rows)
        client = _StubClient(
            _result({"cand_0": _score(2), "cand_1": _score(5), "cand_2": _score(4)})
        )

        output = await jev_search("q", None, _config(), max_results=3, client=client)

        assert [entry["entry_id"] for entry, _score_, *_ in output.entries] == ["e1", "e2", "e0"]

    @pytest.mark.asyncio
    async def test_the_pool_is_larger_than_the_answer(self, stub_lexical):
        calls = stub_lexical(_rows(30))
        client = _StubClient(_result({f"cand_{index}": _score(3) for index in range(30)}))

        output = await jev_search(
            "q", None, _config(candidate_limit=30), max_results=5, client=client
        )

        assert calls[0]["max_results"] == 30
        assert len(output.entries) == 5

    @pytest.mark.asyncio
    async def test_a_pool_smaller_than_the_answer_is_widened_to_it(self, stub_lexical):
        calls = stub_lexical(_rows(4))
        client = _StubClient(_result({f"cand_{index}": _score(3) for index in range(4)}))

        await jev_search("q", None, _config(candidate_limit=2), max_results=10, client=client)

        assert calls[0]["max_results"] == 10

    @pytest.mark.asyncio
    async def test_reranking_never_adds_an_entry_the_lexical_stage_missed(self, stub_lexical):
        stub_lexical(_rows(2))
        client = _StubClient(_result({f"cand_{index}": _score(5) for index in range(5)}))

        output = await jev_search("q", None, _config(), max_results=10, client=client)

        assert len(output.entries) == 2

    @pytest.mark.asyncio
    async def test_rerank_false_asks_nothing_and_keeps_the_lexical_order(self, stub_lexical):
        stub_lexical(_rows(3))
        client = _StubClient(_result({}))

        output = await jev_search("q", None, _config(), max_results=3, rerank=False, client=client)

        assert [entry["entry_id"] for entry, *_ in output.entries] == ["e0", "e1", "e2"]
        assert client.state is None

    @pytest.mark.asyncio
    async def test_the_relevance_floor_drops_what_the_model_rejected(self, stub_lexical):
        stub_lexical(_rows(3))
        client = _StubClient(
            _result({"cand_0": _score(5), "cand_1": _score(1), "cand_2": _score(4)})
        )

        output = await jev_search(
            "q", None, _config(), max_results=3, min_relevance=0.5, client=client
        )

        assert [entry["entry_id"] for entry, *_ in output.entries] == ["e0", "e2"]

    @pytest.mark.asyncio
    async def test_a_pool_rejected_wholesale_is_shown_rather_than_hidden(self, stub_lexical):
        stub_lexical(_rows(3))
        client = _StubClient(_result({f"cand_{index}": _score(1) for index in range(3)}))

        output = await jev_search(
            "q", None, _config(), max_results=3, min_relevance=0.9, client=client
        )

        assert len(output.entries) == 3

    @pytest.mark.asyncio
    async def test_a_confident_recency_reading_lifts_the_newest_entry(self, stub_lexical):
        # e2 is both the worst lexical rank and the newest entry.
        stub_lexical(_rows(3))
        answers: dict[str, JevAnswer] = {f"cand_{index}": _score(3) for index in range(3)}
        answers["query_order"] = JevAnswer(type="choice", value="recency", confidence=0.95)
        client = _StubClient(_result(answers))

        output = await jev_search("q", None, _config(), max_results=3, client=client)

        assert output.entries[0][0]["entry_id"] == "e2"

    @pytest.mark.asyncio
    async def test_an_unsure_reading_changes_nothing(self, stub_lexical):
        stub_lexical(_rows(3))
        answers: dict[str, JevAnswer] = {f"cand_{index}": _score(3) for index in range(3)}
        answers["query_order"] = JevAnswer(type="choice", value="recency", confidence=0.1)
        client = _StubClient(_result(answers))

        output = await jev_search("q", None, _config(), max_results=3, client=client)

        assert output.entries[0][0]["entry_id"] == "e0"

    @pytest.mark.asyncio
    async def test_a_completed_rerank_reports_what_it_cost(self, stub_lexical):
        stub_lexical(_rows(2))
        client = _StubClient(_result({"cand_0": _score(5), "cand_1": _score(2)}))

        output = await jev_search("q", None, _config(), max_results=2, client=client)

        messages = [d.message for d in output.diagnostics]
        assert any("151 ms" in message and "7800 input tokens" in message for message in messages)
        assert all(d.level is DiagnosticLevel.INFO for d in output.diagnostics)


class TestDegradation:
    """Every way the endpoint can fail lands on the lexical ranking."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("client", "expected"),
        [
            (_StubClient(None, available=False), "no API key"),
            (_StubClient(JevError("System One returned HTTP 529: busy")), "HTTP 529"),
            (_StubClient(TimeoutError("timed out")), "unexpected failure"),
            (_StubClient(JevResult(answers={}, model="jev-latest")), "no relevance answers"),
        ],
    )
    async def test_the_fast_ranking_is_shown_with_a_warning(
        self, stub_lexical, client: _StubClient, expected: str
    ):
        stub_lexical(_rows(3))

        output = await jev_search("q", None, _config(), max_results=3, client=client)

        assert [entry["entry_id"] for entry, *_ in output.entries] == ["e0", "e1", "e2"]
        warning = next(d for d in output.diagnostics if d.level is DiagnosticLevel.WARNING)
        # The panel's existing two-phase status line watches this category.
        assert warning.category == "rerank"
        assert expected in warning.message

    @pytest.mark.asyncio
    async def test_an_empty_lexical_pool_asks_nothing(self, stub_lexical):
        stub_lexical([])
        client = _StubClient(_result({}))

        output = await jev_search("q", None, _config(), max_results=3, client=client)

        assert output.entries == []
        assert client.state is None


class TestDescriptors:
    """What the registry, the agent and the search panel see."""

    def test_the_descriptor_routes_the_jev_mode(self):
        descriptor = jev_module.get_tool_descriptor()
        assert (descriptor.name, descriptor.search_mode) == ("jev_search", "jev")
        assert descriptor.needs_embedder is False
        assert descriptor.accepts_expansion is True

    def test_the_panel_is_offered_the_two_switches_the_frontend_reads(self):
        names = [p.name for p in jev_module.get_parameter_descriptors()]
        assert "rerank" in names
        assert "instant_search" in names

    def test_a_malformed_settings_block_still_describes_the_module(self):
        descriptors = jev_module.get_parameter_descriptors(_config(candidate_limit="nope"))
        limit = next(p for p in descriptors if p.name == "candidate_limit")
        assert limit.default == jev_module.DEFAULT_CANDIDATE_LIMIT


class TestClientReuse:
    """A per-keystroke search bar must not open a pool per keystroke."""

    @pytest.mark.asyncio
    async def test_two_searches_share_one_client(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(jev_module, "_CLIENTS", {})
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        settings = JevSearchSettings.from_ariel_config(None).to_client_settings()

        first = jev_module._shared_client(settings)
        second = jev_module._shared_client(settings)

        assert first is second

    @pytest.mark.asyncio
    async def test_a_different_endpoint_gets_its_own_client(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(jev_module, "_CLIENTS", {})
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        default = JevSearchSettings.from_ariel_config(None).to_client_settings()
        gateway = JevSearchSettings.from_ariel_config(
            _config(endpoint="https://gateway.example/v1/systemone")
        ).to_client_settings()

        assert jev_module._shared_client(default) is not jev_module._shared_client(gateway)

    @pytest.mark.asyncio
    async def test_a_client_from_an_ended_loop_is_not_handed_out_again(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(jev_module, "_CLIENTS", {})
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        settings = JevSearchSettings.from_ariel_config(None).to_client_settings()
        # An entry left behind by a loop that has since ended.
        jev_module._CLIENTS[(settings, -1)] = object()  # type: ignore[assignment]

        client = jev_module._shared_client(settings)

        assert list(jev_module._CLIENTS.values()) == [client]
