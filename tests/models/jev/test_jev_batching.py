"""Batching inside ``JevClient.decide``: splitting, refusal, bisection and the concurrency cap.

Every test runs offline on the scripted fake transport and its fake clock; the
injected ``sleep`` only advances that clock. All data is synthetic.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from tests.models.jev._jev_fake_transport import FakeJev, Reply

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV, JevSettings
from osprey.models.jev.errors import (
    DecisionError,
    DecisionRequestError,
    DecisionUnavailableError,
    IncompleteAnswersError,
)
from osprey.models.jev.models import JevRequest
from osprey.models.jev.tokens import estimate_question, estimate_request

SYNTHETIC_KEY = "synthetic-key-not-real"
TYPESAFE_BASE = "https://synthetic-typesafe.invalid"
STATE = "synthetic state: find the warm widget"
NOUL_ANSWER = {"type": "noul", "noul": 0.5}


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (TYPESAFE_KEY_ENV, OPENROUTER_KEY_ENV, "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _questions(count: int) -> dict[str, dict[str, Any]]:
    """``count`` synthetic noul questions of identical size (ids q1..qN, N < 10)."""
    return {
        f"q{i}": {"type": "noul", "instructions": f"Is synthetic widget {i} warm?"}
        for i in range(1, count + 1)
    }


def _typed(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    return dict(
        JevRequest.model_validate(
            {"model": "synthetic", "state": state, "questions": questions}
        ).questions
    )


def _base_tokens(state: Any = STATE) -> int:
    """The estimate of a request carrying no question."""
    return estimate_request(state, {})


def _question_tokens(questions: dict[str, Any], qid: str, state: Any = STATE) -> int:
    typed = _typed(state, questions)
    return estimate_question(qid, typed[qid])


def _config(
    *,
    max_request_tokens: int | None = None,
    max_state_plus_question_tokens: int | None = None,
    max_concurrency: int | None = None,
) -> dict:
    route: dict[str, Any] = {"base_url": TYPESAFE_BASE, "api_key": SYNTHETIC_KEY}
    if max_request_tokens is not None:
        route["max_request_tokens"] = max_request_tokens
    if max_state_plus_question_tokens is not None:
        route["max_state_plus_question_tokens"] = max_state_plus_question_tokens
    jev: dict[str, Any] = {"routes": {"typesafe": route}}
    if max_concurrency is not None:
        jev["max_concurrency"] = max_concurrency
    return {"api": {"jev": jev}}


def _client(fake: FakeJev, **config_kwargs: Any) -> JevClient:
    config = _config(**config_kwargs)
    return JevClient(
        JevSettings.from_config(config),
        config,
        transport=fake.transport,
        clock=fake.clock,
        sleep=fake.sleep,
    )


def _answers_for(questions: dict[str, Any]) -> dict[str, Any]:
    return dict.fromkeys(questions, NOUL_ANSWER)


def _two_per_chunk_limits(questions: dict[str, Any]) -> dict[str, int]:
    """Limits under which exactly two of the equal-size questions fit one request."""
    per_question = _question_tokens(questions, "q1")
    base = _base_tokens()
    return {
        "max_request_tokens": base + 2 * per_question,
        "max_state_plus_question_tokens": base + per_question,
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_questions_over_the_request_budget_split_into_same_state_chunks() -> None:
    questions = _questions(5)
    limits = _two_per_chunk_limits(questions)
    fake = FakeJev(_answers_for(questions))
    with _client(fake, max_concurrency=1, **limits) as client:
        result = client.decide(STATE, questions, budget="batch")

    assert fake.call_count == 3
    sizes = sorted(len(body["questions"]) for body in fake.requests)
    assert sizes == [1, 2, 2]
    assert all(body["state"] == STATE for body in fake.requests)
    sent = [qid for body in fake.requests for qid in body["questions"]]
    assert sorted(sent) == sorted(questions)
    for body in fake.requests:
        chunk = _typed(STATE, body["questions"])
        assert estimate_request(STATE, chunk) <= limits["max_request_tokens"]
    assert set(result.answers) == set(questions)
    assert len(result.calls) == 3


def test_no_question_count_cap_when_everything_fits() -> None:
    questions = _questions(9)
    fake = FakeJev(_answers_for(questions))
    with _client(fake) as client:
        result = client.decide(STATE, questions, budget="batch")
    assert fake.call_count == 1
    assert len(result.calls) == 1
    assert set(result.answers) == set(questions)


def test_a_lone_question_at_the_limit_is_sent_alone() -> None:
    questions = _questions(3)
    questions["q9"] = {
        "type": "noul",
        "instructions": "Is the synthetic widget warm? " + "synthetic detail " * 40,
    }
    base = _base_tokens()
    big = _question_tokens(questions, "q9")
    small = _question_tokens(questions, "q1")
    assert big > 3 * small
    limit = base + big
    fake = FakeJev(_answers_for(questions))
    with _client(
        fake, max_concurrency=1, max_request_tokens=limit, max_state_plus_question_tokens=limit
    ) as client:
        result = client.decide(STATE, questions, budget="batch")

    assert set(result.answers) == set(questions)
    groups = sorted(sorted(body["questions"]) for body in fake.requests)
    assert ["q9"] in groups
    assert len(groups) == 2
    assert sorted(q for g in groups for q in g) == sorted(questions)


def test_missing_id_in_one_chunk_raises_incomplete_answers() -> None:
    questions = _questions(5)
    limits = _two_per_chunk_limits(questions)
    answers = _answers_for(questions)
    del answers["q5"]
    fake = FakeJev(answers)
    with _client(fake, max_concurrency=1, **limits) as client:
        with pytest.raises(IncompleteAnswersError) as info:
            client.decide(STATE, questions, budget="batch")
    assert "q5" in info.value.message
    # Every billed chunk, the incomplete one included, is accounted for.
    assert len(info.value.completed_calls) == 3


# ---------------------------------------------------------------------------
# Local refusal
# ---------------------------------------------------------------------------


def test_oversized_state_is_refused_without_http() -> None:
    questions = _questions(2)
    big_state = "synthetic state " * 200
    fake = FakeJev(_answers_for(questions))
    with _client(fake, max_request_tokens=400, max_state_plus_question_tokens=400) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(big_state, questions, budget="batch")
    assert info.value.error_type == "state_too_large"
    assert fake.call_count == 0


# ---------------------------------------------------------------------------
# Bisection
# ---------------------------------------------------------------------------


def _multi(body: dict[str, Any]) -> bool:
    return len(body["questions"]) > 1


def test_413_on_a_multi_question_chunk_bisects_and_completes() -> None:
    questions = _questions(2)
    fake = FakeJev(_answers_for(questions)).reject_if(_multi, status=413)
    with _client(fake) as client:
        result = client.decide(STATE, questions, budget="batch")
    assert fake.call_count == 3
    assert set(result.answers) == set(questions)
    assert len(result.calls) == 2


def test_max_tokens_exceeded_bisects_and_completes() -> None:
    questions = _questions(2)
    body = {"detail": {"error_type": "max_tokens_exceeded", "message": "synthetic: too long"}}
    fake = FakeJev(_answers_for(questions)).reject_if(_multi, status=400, body=body)
    with _client(fake) as client:
        result = client.decide(STATE, questions, budget="batch")
    assert fake.call_count == 3
    assert set(result.answers) == set(questions)


def test_bisection_happens_once() -> None:
    questions = _questions(4)
    fake = FakeJev(_answers_for(questions)).reject_if(_multi, status=413)
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, questions, budget="batch")
    assert info.value.status == 413
    # The chunk and its two halves; a half is never bisected again.
    assert fake.call_count <= 3


def test_413_on_a_single_question_raises() -> None:
    questions = _questions(1)
    fake = FakeJev(_answers_for(questions), responses=[413])
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, questions, budget="batch")
    assert info.value.status == 413
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# Failure after billed chunks
# ---------------------------------------------------------------------------


def test_second_chunk_failing_carries_the_first_chunks_call() -> None:
    questions = _questions(4)
    limits = _two_per_chunk_limits(questions)
    fake = FakeJev(_answers_for(questions), responses=[None, Reply(status=400)])
    with _client(fake, max_concurrency=1, **limits) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, questions, budget="batch")
    assert fake.call_count == 2
    (call,) = info.value.completed_calls
    assert call.request_id == "synthetic-req-1"


# ---------------------------------------------------------------------------
# Deadline and concurrency
# ---------------------------------------------------------------------------


def test_interactive_deadline_is_shared_by_every_chunk() -> None:
    questions = _questions(2)
    per_question = _question_tokens(questions, "q1")
    limit = _base_tokens() + per_question
    fake = FakeJev(_answers_for(questions), latency=1.6)
    start = fake.clock()
    with _client(
        fake, max_concurrency=1, max_request_tokens=limit, max_state_plus_question_tokens=limit
    ) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, questions, budget="interactive")
    assert fake.clock() - start <= 3.0
    # The first chunk was billed before the second ran out of time.
    assert len(info.value.completed_calls) == 1


def test_a_chunk_after_the_deadline_is_not_sent() -> None:
    questions = _questions(2)
    per_question = _question_tokens(questions, "q1")
    limit = _base_tokens() + per_question
    # The first chunk alone uses the whole interactive budget.
    fake = FakeJev(_answers_for(questions), responses=[Reply(latency=1.0)], latency=0.0)
    with _client(
        fake, max_concurrency=1, max_request_tokens=limit, max_state_plus_question_tokens=limit
    ) as client:
        deadline = fake.clock() + 1.0
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, questions, budget="interactive", deadline=deadline)
    assert info.value.error_type == "deadline_exceeded"
    assert fake.call_count == 1


def test_in_flight_requests_never_exceed_max_concurrency_across_threads() -> None:
    questions = _questions(8)
    limit = _base_tokens() + 2 * _question_tokens(questions, "q1")
    fake = FakeJev(_answers_for(questions), dwell=0.05)
    errors: list[BaseException] = []
    with _client(
        fake,
        max_concurrency=2,
        max_request_tokens=limit,
        max_state_plus_question_tokens=limit,
    ) as client:

        def run() -> None:
            try:
                result = client.decide(STATE, questions, budget="batch")
                assert len(result.calls) == 4
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert not errors
    assert fake.call_count == 8
    assert fake.peak_in_flight <= 2


def test_every_escape_is_a_decision_error() -> None:
    questions = _questions(3)
    limits = _two_per_chunk_limits(questions)
    fake = FakeJev(_answers_for(questions), responses=[500, 500, 500, 500, 500, 500])
    with _client(fake, max_concurrency=1, **limits) as client:
        with pytest.raises(DecisionError):
            client.decide(STATE, questions, budget="interactive")
