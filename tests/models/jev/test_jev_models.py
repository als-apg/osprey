"""Typed Jev requests, answers, the answer normaliser and the call records."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from osprey.models.jev.errors import IncompleteAnswersError
from osprey.models.jev.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevAttempt,
    JevCall,
    JevRequest,
    JevResponse,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    parse_response,
)

MODEL = "typesafe/jev-1.13"
SERVED = "typesafe/jev-1.13-20260917"


def _request(**questions: dict) -> JevRequest:
    return JevRequest.model_validate(
        {"model": MODEL, "state": "some state", "questions": questions}
    )


def _payload(answers: dict, *, cost: float | None = None) -> dict:
    usage: dict = {"input_tokens": 476, "output_tokens": 70}
    if cost is not None:
        usage["cost"] = cost
    return {"model": SERVED, "answers": answers, "usage": usage}


CHOICE_ANSWER = {
    "type": "choice",
    "choice": "payments",
    "confidence": 0.67,
    "probabilities": {"payments": 0.78, "frontend": 0.22, "account": 0},
}
SCORE_ANSWER = {
    "type": "score",
    "score": 1.99,
    "confidence": 0.99,
    "probabilities": {"0": 0, "1": 0, "2": 1},
    "legend": {"0": "low", "1": "mid", "2": "high"},
}
NOUL_ANSWER = {"type": "noul", "noul": 0.96}


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def test_request_accepts_all_three_question_types() -> None:
    req = _request(
        is_bug={
            "type": "noul",
            "instructions": "Is it a bug?",
            "criteria": {"true": "broken", "false": "a question"},
        },
        team={"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": "B"}},
        urgency={
            "type": "score",
            "instructions": {"question": "How urgent?"},
            "criteria": ["lo", "hi"],
        },
    )
    assert isinstance(req.questions["is_bug"], NoulQuestion)
    assert isinstance(req.questions["team"], ChoiceQuestion)
    assert isinstance(req.questions["urgency"], ScoreQuestion)


def test_noul_without_criteria_is_accepted() -> None:
    req = _request(q={"type": "noul", "instructions": "Is it?"})
    assert req.questions["q"].criteria is None


@pytest.mark.parametrize("criteria", [{"true": "yes"}, {"false": "no"}, {}])
def test_noul_with_one_sided_criteria_is_refused(criteria: dict) -> None:
    with pytest.raises(ValidationError):
        _request(q={"type": "noul", "instructions": "Is it?", "criteria": criteria})


@pytest.mark.parametrize("instructions", ["", "   ", {}, [], None])
@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul"},
        {"type": "choice", "criteria": {"a": None}},
        {"type": "score", "criteria": ["lo", "hi"]},
    ],
)
def test_empty_instructions_are_refused_on_every_type(question: dict, instructions: object) -> None:
    with pytest.raises(ValidationError):
        _request(q={**question, "instructions": instructions})


def test_missing_instructions_are_refused() -> None:
    with pytest.raises(ValidationError):
        _request(q={"type": "noul"})


def test_choice_option_count_bounds() -> None:
    _request(
        q={"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(255)}}
    )
    with pytest.raises(ValidationError):
        _request(
            q={
                "type": "choice",
                "instructions": "x",
                "criteria": {f"o{i}": None for i in range(256)},
            }
        )
    with pytest.raises(ValidationError):
        _request(q={"type": "choice", "instructions": "x", "criteria": {}})


def test_score_level_count_bounds() -> None:
    _request(q={"type": "score", "instructions": "x", "criteria": [str(i) for i in range(10)]})
    with pytest.raises(ValidationError):
        _request(q={"type": "score", "instructions": "x", "criteria": ["only"]})
    with pytest.raises(ValidationError):
        _request(q={"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]})


def test_unknown_question_type_is_refused() -> None:
    with pytest.raises(ValidationError):
        _request(q={"type": "rank", "instructions": "x"})


def test_request_needs_at_least_one_question() -> None:
    with pytest.raises(ValidationError):
        JevRequest.model_validate({"model": MODEL, "state": "s", "questions": {}})


# ---------------------------------------------------------------------------
# Answers and the normaliser
# ---------------------------------------------------------------------------


def test_published_response_parses_into_typed_answers() -> None:
    resp = parse_response(
        _payload(
            {"is_bug": NOUL_ANSWER, "team": CHOICE_ANSWER, "urgency": SCORE_ANSWER}, cost=2e-5
        ),
        ["is_bug", "team", "urgency"],
    )
    assert isinstance(resp.answers["is_bug"], NoulAnswer)
    assert isinstance(resp.answers["team"], ChoiceAnswer)
    assert isinstance(resp.answers["urgency"], ScoreAnswer)
    assert resp.model == SERVED
    assert resp.usage.input_tokens == 476
    assert resp.usage.cost == pytest.approx(2e-5)


def test_noul_answer_has_no_confidence() -> None:
    assert "confidence" not in NoulAnswer.model_fields


def test_choice_without_probabilities_or_confidence_is_accepted() -> None:
    ans = ChoiceAnswer.model_validate({"type": "choice", "choice": "a"})
    assert ans.probabilities is None
    assert ans.confidence is None


def test_score_without_probabilities_or_confidence_is_accepted() -> None:
    ans = ScoreAnswer.model_validate(
        {"type": "score", "score": 0.4, "legend": {"0": "lo", "1": "hi"}}
    )
    assert ans.probabilities is None
    assert ans.confidence is None


def test_score_is_a_zero_based_expected_index_with_string_legend() -> None:
    ans = ScoreAnswer.model_validate(SCORE_ANSWER)
    assert 0.0 <= ans.score <= len(ans.legend) - 1
    assert set(ans.legend) == {"0", "1", "2"}


def test_unrenormalised_sum_is_accepted() -> None:
    ans = ChoiceAnswer.model_validate(
        {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.49}, "confidence": 0.1}
    )
    assert ans.probabilities is not None
    assert sum(ans.probabilities.values()) == pytest.approx(0.99)


def test_float_noise_normalises_equal() -> None:
    a = ChoiceAnswer.model_validate({"type": "choice", "choice": "x", "probabilities": {"x": 0.57}})
    b = ChoiceAnswer.model_validate(
        {"type": "choice", "choice": "x", "probabilities": {"x": 0.5700000000000001}}
    )
    assert a.probabilities == b.probabilities


@pytest.mark.parametrize("small", [0.0, 0.004, 0])
def test_tiny_choice_option_is_absent_after_normalisation(small: float) -> None:
    ans = ChoiceAnswer.model_validate(
        {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": small, "c": 0.005}}
    )
    assert ans.probabilities is not None
    assert "b" not in ans.probabilities
    assert "c" in ans.probabilities


def test_choice_probabilities_are_rounded_to_three_places() -> None:
    ans = ChoiceAnswer.model_validate(
        {"type": "choice", "choice": "a", "probabilities": {"a": 0.123456}}
    )
    assert ans.probabilities is not None
    assert ans.probabilities["a"] == round(0.123456, 3)


def test_noul_is_rounded_to_three_places() -> None:
    a = NoulAnswer.model_validate({"type": "noul", "noul": 0.9600000000000001})
    b = NoulAnswer.model_validate({"type": "noul", "noul": 0.96})
    assert a.noul == b.noul
    assert NoulAnswer.model_validate({"type": "noul", "noul": 0.123456}).noul == round(0.123456, 3)


def test_normaliser_runs_at_response_parse() -> None:
    resp = parse_response(
        _payload(
            {
                "team": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.5700000000000001, "b": 0.0, "c": 0.43},
                }
            }
        ),
        ["team"],
    )
    probs = resp.answers["team"].probabilities  # type: ignore[union-attr]
    assert probs == {"a": 0.57, "c": 0.43}


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_missing_requested_id_raises_incomplete_answers() -> None:
    with pytest.raises(IncompleteAnswersError) as info:
        parse_response(_payload({"team": CHOICE_ANSWER}), ["team", "is_bug"])
    assert "is_bug" in info.value.message


def test_all_ids_present_passes() -> None:
    resp = parse_response(_payload({"team": CHOICE_ANSWER}), ["team"])
    assert set(resp.answers) == {"team"}


def test_response_without_requested_ids_skips_completeness() -> None:
    resp = JevResponse.model_validate(_payload({"team": CHOICE_ANSWER}))
    assert "team" in resp.answers


def test_openrouter_extra_fields_are_tolerated() -> None:
    payload = _payload({"team": CHOICE_ANSWER}, cost=1e-5)
    payload["id"] = "gen-dec-1-abc"
    payload["provider"] = "TypeSafe"
    resp = parse_response(payload, ["team"])
    assert resp.id == "gen-dec-1-abc"


def test_direct_route_has_no_cost() -> None:
    resp = parse_response(_payload({"team": CHOICE_ANSWER}), ["team"])
    assert resp.usage.cost is None


# ---------------------------------------------------------------------------
# Call records
# ---------------------------------------------------------------------------


def _call(**overrides: object) -> JevCall:
    fields: dict = {
        "input_tokens": 100,
        "output_tokens": 10,
        "billed_cost_usd": None,
        "cost_usd": 4.2e-6,
        "cost_source": "computed",
        "list_cost_usd": 4.2e-6,
        "latency_s": 0.3,
        "served_model": SERVED,
        "request_id": "req-1",
        "attempts": 1,
    }
    fields.update(overrides)
    return JevCall(**fields)


def test_jev_call_fields() -> None:
    names = {f.name for f in dataclasses.fields(JevCall)}
    assert names == {
        "input_tokens",
        "output_tokens",
        "billed_cost_usd",
        "cost_usd",
        "cost_source",
        "list_cost_usd",
        "latency_s",
        "served_model",
        "request_id",
        "attempts",
    }


def test_jev_call_both_cost_sources() -> None:
    billed = _call(billed_cost_usd=2e-5, cost_usd=2e-5, cost_source="billed")
    computed = _call()
    assert billed.cost_source == "billed"
    assert computed.cost_source == "computed"
    assert computed.billed_cost_usd is None


def test_jev_call_refuses_unknown_cost_source() -> None:
    with pytest.raises(ValueError):
        _call(cost_source="litellm")


def test_jev_result_shape() -> None:
    ans = NoulAnswer.model_validate(NOUL_ANSWER)
    result = JevResult(
        answers={"q": ans},
        calls=[_call()],
        requested_model=MODEL,
        served_model=SERVED,
        route="openrouter",
    )
    assert result.answers["q"] is ans
    assert len(result.calls) == 1
    assert result.route == "openrouter"


def test_jev_attempt_is_frozen_and_bodiless() -> None:
    attempt = JevAttempt(
        chunk_index=0,
        attempt=1,
        status=None,
        error_type="timeout",
        latency_s=10.0,
        input_tokens=None,
        output_tokens=None,
        billed_cost_usd=None,
        served_model=None,
        request_id=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.status = 200  # type: ignore[misc]
    names = {f.name for f in dataclasses.fields(JevAttempt)}
    assert names == {
        "chunk_index",
        "attempt",
        "status",
        "error_type",
        "latency_s",
        "input_tokens",
        "output_tokens",
        "billed_cost_usd",
        "served_model",
        "request_id",
    }
    assert not any("body" in n for n in names)
