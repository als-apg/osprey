"""Tests for the TypeSafe System One ("Jev") client.

Every test drives a stub transport rather than the network: what is under test
is the protocol this client speaks and the retry rule it applies, neither of
which needs a real endpoint to be wrong.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from osprey.services.jev import (
    JevClient,
    JevError,
    JevSettings,
    JevUnavailableError,
    choice_question,
    noul_question,
    score_question,
)


class _Response:
    """The parts of an ``httpx.Response`` the client reads."""

    def __init__(self, status_code: int, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _Transport:
    """A stub ``httpx.AsyncClient`` that replays a scripted list of outcomes."""

    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, *, json: Any, headers: dict[str, str]) -> _Response:
        self.requests.append({"url": url, "json": json, "headers": headers})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None


def _client(outcomes: list[Any], **settings: Any) -> tuple[JevClient, _Transport]:
    """Build a client wired to a stub transport and a fixed key."""
    transport = _Transport(outcomes)
    client = JevClient(JevSettings(**settings), api_key="test-key")
    client._client = transport
    return client, transport


def _answers(**answers: Any) -> dict[str, Any]:
    """Build a well-formed response body."""
    return {
        "model": "jev-latest",
        "answers": answers,
        "usage": {"input_tokens": 7800, "output_tokens": 120},
    }


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch):
    """Retries are tested for their count, not for how long they wait."""

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("osprey.services.jev.client.asyncio.sleep", _instant)


class TestQuestionBuilders:
    """The three question shapes the endpoint accepts."""

    def test_score_question_carries_its_levels_in_order(self):
        question = score_question("How relevant?", ["no", "maybe", "yes"])
        assert question == {
            "type": "score",
            "instructions": "How relevant?",
            "criteria": ["no", "maybe", "yes"],
        }

    def test_score_question_refuses_a_single_level(self):
        with pytest.raises(ValueError, match="at least two"):
            score_question("How relevant?", ["only"])

    def test_choice_question_refuses_a_single_option(self):
        with pytest.raises(ValueError, match="at least two"):
            choice_question("Which?", {"only": "the only one"})

    def test_noul_question_omits_absent_criteria(self):
        assert noul_question("Is this a fault report?") == {
            "type": "noul",
            "instructions": "Is this a fault report?",
        }


class TestAvailability:
    """A deployment with no key is a state, not an error."""

    def test_a_client_without_a_key_is_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        assert JevClient(JevSettings()).is_available() is False

    def test_the_key_is_read_from_the_configured_variable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FACILITY_JEV_KEY", "k")
        client = JevClient(JevSettings(api_key_env="FACILITY_JEV_KEY"))
        assert client.is_available() is True

    @pytest.mark.asyncio
    async def test_asking_without_a_key_never_opens_a_socket(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        client = JevClient(JevSettings())
        with pytest.raises(JevUnavailableError, match="TYPESAFE_API_KEY"):
            await client.ask({"q": "x"}, {"a": noul_question("true?")})
        assert client._client is None


class TestAsk:
    """One request, one state, many typed answers."""

    @pytest.mark.asyncio
    async def test_one_request_carries_the_state_and_every_question(self):
        client, transport = _client([_Response(200, _answers(a={"type": "noul", "noul": 0.9}))])

        await client.ask(
            {"query": "beam loss"},
            {"a": noul_question("Is it about beam loss?"), "b": noul_question("Recent?")},
        )

        assert len(transport.requests) == 1
        body = transport.requests[0]["json"]
        assert body["state"] == {"query": "beam loss"}
        assert set(body["questions"]) == {"a", "b"}
        assert body["model"] == "jev-latest"
        assert transport.requests[0]["headers"]["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_each_answer_type_is_normalized_onto_one_shape(self):
        client, _ = _client(
            [
                _Response(
                    200,
                    _answers(
                        s={
                            "type": "score",
                            "score": 4,
                            "legend": {"1": "no", "5": "yes"},
                            "confidence": 0.8,
                        },
                        c={"type": "choice", "choice": "recency", "confidence": 0.7},
                        n={"type": "noul", "noul": 0.25},
                    ),
                )
            ]
        )

        result = await client.ask({"x": 1}, {"s": noul_question("q")})

        assert result.answers["s"].value == 4
        assert result.answers["s"].legend == {"1": "no", "5": "yes"}
        assert result.answers["c"].value == "recency"
        assert result.answers["c"].confidence == 0.7
        # A noul's value IS a probability, so it carries no separate confidence.
        assert result.answers["n"].value == 0.25
        assert result.answers["n"].confidence is None

    @pytest.mark.asyncio
    async def test_usage_is_reported_so_a_caller_can_price_a_search(self):
        client, _ = _client([_Response(200, _answers(a={"type": "noul", "noul": 1.0}))])
        result = await client.ask({"x": 1}, {"a": noul_question("q")})
        assert (result.input_tokens, result.output_tokens) == (7800, 120)

    @pytest.mark.asyncio
    async def test_a_reply_without_answers_is_refused(self):
        client, _ = _client([_Response(200, {"model": "jev-latest"})])
        with pytest.raises(JevError, match="no 'answers' map"):
            await client.ask({"x": 1}, {"a": noul_question("q")})

    @pytest.mark.asyncio
    async def test_an_unknown_answer_type_is_refused_rather_than_guessed_at(self):
        client, _ = _client([_Response(200, _answers(a={"type": "vibes", "vibes": 1}))])
        with pytest.raises(JevError, match="Unknown answer type"):
            await client.ask({"x": 1}, {"a": noul_question("q")})

    @pytest.mark.asyncio
    async def test_asking_nothing_is_a_caller_mistake(self):
        client, _ = _client([])
        with pytest.raises(ValueError, match="at least one question"):
            await client.ask({"x": 1}, {})


class TestRetries:
    """Only the two statuses that mean "later" are worth a second attempt."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 529])
    async def test_a_later_status_is_retried(self, status: int):
        client, transport = _client(
            [
                _Response(status, text="slow down"),
                _Response(200, _answers(a={"type": "noul", "noul": 1.0})),
            ]
        )

        result = await client.ask({"x": 1}, {"a": noul_question("q")})

        assert len(transport.requests) == 2
        assert result.answers["a"].value == 1.0

    @pytest.mark.asyncio
    async def test_a_request_error_is_not_retried(self):
        client, transport = _client([_Response(422, text="bad question")])

        with pytest.raises(JevError, match="HTTP 422"):
            await client.ask({"x": 1}, {"a": noul_question("q")})

        assert len(transport.requests) == 1

    @pytest.mark.asyncio
    async def test_retries_are_bounded_by_the_configured_count(self):
        client, transport = _client([_Response(429, text="no")] * 4, max_retries=2)

        with pytest.raises(JevError, match="HTTP 429"):
            await client.ask({"x": 1}, {"a": noul_question("q")})

        assert len(transport.requests) == 3

    @pytest.mark.asyncio
    async def test_a_transport_failure_that_never_clears_reads_as_unavailable(self):
        client, transport = _client(
            [httpx.ConnectError("no route"), httpx.ConnectError("no route")], max_retries=1
        )

        with pytest.raises(JevUnavailableError, match="Could not reach"):
            await client.ask({"x": 1}, {"a": noul_question("q")})

        assert len(transport.requests) == 2

    @pytest.mark.asyncio
    async def test_a_transport_failure_that_clears_is_not_surfaced(self):
        client, _ = _client(
            [
                httpx.ConnectError("no route"),
                _Response(200, _answers(a={"type": "noul", "noul": 0.5})),
            ]
        )

        result = await client.ask({"x": 1}, {"a": noul_question("q")})

        assert result.answers["a"].value == 0.5
