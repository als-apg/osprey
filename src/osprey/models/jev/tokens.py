"""Conservative token estimates for Jev requests, computed before any HTTP call.

The estimate is compared against a route's ``max_request_tokens`` and
``max_state_plus_question_tokens``, so it must never undercount a real request
body. Text is charged at 4 characters per token, below the lowest ratio
measured on long prose (about 4.2), and every choice option or score level
carries a fixed per-option cost on top of its text.

* request: a fixed overhead plus the state plus every question;
* noul: the characters of its JSON;
* choice: 18 tokens per option plus the option's text, plus its instructions;
* score: sized like a choice over its levels.

Any other question type raises ``TypeError``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from osprey.models.jev.models import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

__all__ = [
    "CHARS_PER_TOKEN",
    "OPTION_OVERHEAD_TOKENS",
    "REQUEST_OVERHEAD_TOKENS",
    "estimate_question",
    "estimate_request",
    "estimate_state_plus_longest",
]

# Fixed tokens every request costs beyond its state and questions.
REQUEST_OVERHEAD_TOKENS = 280
# Tokens each choice option or score level costs beyond its own text.
OPTION_OVERHEAD_TOKENS = 18
# Characters charged per token; below every measured ratio, so text is overcounted.
CHARS_PER_TOKEN = 4


def _tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def _json_chars(value: Any) -> int:
    """Length of ``value`` as JSON; a plain string counts with its quotes."""
    return len(json.dumps(value, ensure_ascii=False))


def _state_tokens(state: Any) -> int:
    if isinstance(state, str):
        return _tokens(len(state))
    return _tokens(_json_chars(state))


def _options_tokens(options: list[tuple[str, Any]], instructions: Any) -> int:
    total = _tokens(_json_chars(instructions))
    for label, text in options:
        total += OPTION_OVERHEAD_TOKENS + _tokens(_json_chars(label) + _json_chars(text))
    return total


def estimate_question(question_id: str, question: Question) -> int:
    """Tokens one question adds to a request, its id included."""
    id_tokens = _tokens(_json_chars(question_id))
    if isinstance(question, NoulQuestion):
        return id_tokens + _tokens(len(question.model_dump_json(by_alias=True)))
    if isinstance(question, ChoiceQuestion):
        options = list(question.criteria.items())
        return id_tokens + _options_tokens(options, question.instructions)
    if isinstance(question, ScoreQuestion):
        levels = [(str(index), text) for index, text in enumerate(question.criteria)]
        return id_tokens + _options_tokens(levels, question.instructions)
    kind = getattr(question, "type", type(question).__name__)
    raise TypeError(f"cannot estimate tokens for question type {kind!r}")


def estimate_request(state: Any, questions: Mapping[str, Question]) -> int:
    """Estimated input tokens of one request: overhead, state and every question."""
    return (
        REQUEST_OVERHEAD_TOKENS
        + _state_tokens(state)
        + sum(estimate_question(qid, q) for qid, q in questions.items())
    )


def estimate_state_plus_longest(state: Any, questions: Mapping[str, Question]) -> int:
    """``estimate_request`` of the one-question request holding the longest question."""
    if not questions:
        raise ValueError("a request needs at least one question")
    longest = max(questions, key=lambda qid: estimate_question(qid, questions[qid]))
    return estimate_request(state, {longest: questions[longest]})
