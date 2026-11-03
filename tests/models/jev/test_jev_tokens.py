"""The Jev request-size estimator: it must never undercount a real request body."""

from __future__ import annotations

import json
import math

import pytest

from osprey.models.jev.config import JevSettings
from osprey.models.jev.models import ChoiceQuestion, JevRequest, NoulQuestion, ScoreQuestion
from osprey.models.jev.tokens import (
    CHARS_PER_TOKEN,
    OPTION_OVERHEAD_TOKENS,
    REQUEST_OVERHEAD_TOKENS,
    estimate_question,
    estimate_request,
    estimate_state_plus_longest,
)

MODEL = "typesafe/jev-1.13"

# Lowest chars-per-token ratio measured on long prose cards (4.18 to 4.42):
# the ratio at which a body holds the most real tokens per character.
WORST_CHARS_PER_TOKEN = 4.18

STATE = "Show me the vacuum pressure readings in sector 3 of the storage plant."

CARD = (
    "Beam position monitor horizontal readback in millimetres, sampled at ten hertz "
    "and averaged over one turn; one channel per monitor, grouped by sector and cell, "
    "used for orbit correction and for trend plots of the closed orbit over a shift. "
    "Distinct from the vertical plane and from the raw button signals."
)


def _body(state: str, questions: dict) -> str:
    request = JevRequest(model=MODEL, state=state, questions=questions)
    return request.model_dump_json(by_alias=True)


def _real_tokens(body: str) -> float:
    return len(body) / WORST_CHARS_PER_TOKEN


def test_constants_match_the_spec() -> None:
    assert REQUEST_OVERHEAD_TOKENS == 280
    assert OPTION_OVERHEAD_TOKENS == 18
    assert CHARS_PER_TOKEN == 4


def test_empty_request_costs_the_fixed_overhead_plus_state() -> None:
    q = NoulQuestion(instructions="x")
    total = estimate_request(STATE, {"q": q})
    assert total == REQUEST_OVERHEAD_TOKENS + math.ceil(len(STATE) / 4) + estimate_question("q", q)


def test_structured_state_is_sized_by_its_json() -> None:
    state = {"query": STATE, "hints": ["a", "b"]}
    q = NoulQuestion(instructions="x")
    expected_state = math.ceil(len(json.dumps(state)) / 4)
    assert estimate_request(state, {"q": q}) - estimate_request("", {"q": q}) == expected_state


def test_noul_is_sized_by_its_json() -> None:
    q = NoulQuestion(instructions=CARD, criteria={"true": "yes it is", "false": "no it is not"})
    assert estimate_question("q", q) >= len(q.model_dump_json(by_alias=True)) / 4


def test_choice_charges_eighteen_tokens_per_option() -> None:
    one = ChoiceQuestion(instructions="pick", criteria={"a": None})
    two = ChoiceQuestion(instructions="pick", criteria={"a": None, "b": None})
    assert estimate_question("q", two) - estimate_question("q", one) >= OPTION_OVERHEAD_TOKENS


def test_choice_near_the_route_limit_cannot_undercount_the_real_body() -> None:
    limit = JevSettings().active_route.max_state_plus_question_tokens
    target = 0.95 * limit
    criteria: dict[str, str | None] = {}
    i = 0
    while True:
        trial = dict(criteria)
        trial[f"kind_{i:03d}"] = f"{CARD} {CARD} Variant {i}."
        question = ChoiceQuestion(instructions="Which kind does the query ask for?", criteria=trial)
        if estimate_state_plus_longest(STATE, {"kind": question}) > target:
            break
        criteria = trial
        i += 1
        assert i < 255, "card too short to reach the target within the option cap"
    question = ChoiceQuestion(instructions="Which kind does the query ask for?", criteria=criteria)
    estimate = estimate_state_plus_longest(STATE, {"kind": question})
    assert 0.9 * limit < estimate <= target

    real = _real_tokens(_body(STATE, {"kind": question}))
    assert estimate >= real
    assert real <= limit


def test_score_estimates_at_least_its_serialised_chars_at_the_worst_ratio() -> None:
    q = ScoreQuestion(
        instructions=CARD,
        criteria=[f"Level {n}: {CARD}" for n in range(5)],
    )
    serialised = len(q.model_dump_json(by_alias=True))
    assert estimate_question("q", q) >= serialised / WORST_CHARS_PER_TOKEN
    assert estimate_request(STATE, {"q": q}) >= _real_tokens(_body(STATE, {"q": q}))


def test_score_charges_like_a_choice_over_its_levels() -> None:
    two = ScoreQuestion(instructions="rate", criteria=["low", "high"])
    three = ScoreQuestion(instructions="rate", criteria=["low", "mid", "high"])
    assert estimate_question("q", three) - estimate_question("q", two) >= OPTION_OVERHEAD_TOKENS


def test_request_sums_every_question_once() -> None:
    a = NoulQuestion(instructions="is it vacuum?")
    b = ChoiceQuestion(instructions="which", criteria={"x": "ex", "y": "why"})
    total = estimate_request(STATE, {"a": a, "b": b})
    assert total == (
        REQUEST_OVERHEAD_TOKENS
        + math.ceil(len(STATE) / 4)
        + estimate_question("a", a)
        + estimate_question("b", b)
    )


def test_state_plus_longest_is_the_one_question_request_of_the_longest() -> None:
    short = NoulQuestion(instructions="is it vacuum?")
    long = ChoiceQuestion(instructions="which", criteria={f"o{i}": CARD for i in range(10)})
    questions = {"short": short, "long": long}
    assert estimate_state_plus_longest(STATE, questions) == estimate_request(STATE, {"long": long})
    assert estimate_state_plus_longest(STATE, questions) < estimate_request(STATE, questions)


def test_state_plus_longest_includes_the_fixed_overhead() -> None:
    q = NoulQuestion(instructions="x")
    assert estimate_state_plus_longest("", {"q": q}) >= REQUEST_OVERHEAD_TOKENS


def test_requests_without_questions_are_refused() -> None:
    with pytest.raises(ValueError):
        estimate_state_plus_longest(STATE, {})


def test_unknown_question_type_raises() -> None:
    class Rank:
        type = "rank"
        instructions = "order these"

    with pytest.raises(TypeError, match="rank"):
        estimate_question("q", Rank())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        estimate_request(STATE, {"q": Rank()})  # type: ignore[dict-item]
