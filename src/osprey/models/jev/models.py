"""Typed Jev requests, answers and per-call records.

Requests are validated locally before any HTTP call: every question carries
non-empty ``instructions``; a noul's ``criteria`` is either absent or names
both ``true`` and ``false``; a choice has 1..255 options; a score has 2..10
ordered levels (index 0 is the low end).

Answers pass through one normaliser at parse, before any consumer sees them,
so every rule, tie-break and calibration record reads the same values:

* choice ``probabilities`` are rounded to 3 decimal places and every option
  below 0.005 is dropped (the service returns every option, zeros included,
  with float noise such as ``0.5700000000000001``);
* a noul value is rounded to 3 decimal places.

Probability sums are never renormalised. A score answer's ``score`` is the
expected 0-based level index; its ``legend`` and ``probabilities`` are keyed
by the level index as a string.

``parse_response`` raises ``IncompleteAnswersError`` when any requested id is
missing from the reply; nothing partial is returned.

``JevCall`` is one record per completed request, ``JevResult`` the value a
decision returns, and ``JevAttempt`` one record per HTTP attempt. None of them
holds a request or response body.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from osprey.models.jev.errors import IncompleteAnswersError

__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "PROBABILITY_DECIMALS",
    "PROBABILITY_FLOOR",
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "CostSource",
    "Guidance",
    "JevAttempt",
    "JevCall",
    "JevRequest",
    "JevResponse",
    "JevResult",
    "JevUsage",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "Question",
    "Route",
    "ScoreAnswer",
    "ScoreQuestion",
    "normalise_choice_probabilities",
    "normalise_noul",
    "parse_response",
]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
PROBABILITY_DECIMALS = 3
PROBABILITY_FLOOR = 0.005

Route = Literal["typesafe", "openrouter"]
CostSource = Literal["billed", "computed"]

# Structured guidance accepted in ``instructions`` and criteria values.
Guidance = str | dict[str, JsonValue] | list[JsonValue]


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


def normalise_choice_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Round to 3 dp and drop every option below the floor; never renormalise."""
    return {
        option: round(float(p), PROBABILITY_DECIMALS)
        for option, p in probabilities.items()
        if float(p) >= PROBABILITY_FLOOR
    }


def normalise_noul(value: float) -> float:
    """Round a noul probability to 3 dp."""
    return round(float(value), PROBABILITY_DECIMALS)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _require_guidance(value: Any) -> Any:
    if value is None:
        raise ValueError("instructions are required")
    if isinstance(value, str) and not value.strip():
        raise ValueError("instructions must not be blank")
    if isinstance(value, (dict, list)) and not value:
        raise ValueError("instructions must not be empty")
    return value


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: Guidance

    @field_validator("instructions", mode="before")
    @classmethod
    def _instructions_non_empty(cls, value: Any) -> Any:
        return _require_guidance(value)


class NoulCriteria(BaseModel):
    """Both sides of a yes/no question; the wire keys are ``true`` and ``false``."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    true: Guidance | None = Field(alias="true")
    false: Guidance | None = Field(alias="false")


class NoulQuestion(_QuestionBase):
    """A yes/no question."""

    type: Literal["noul"] = "noul"
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_QuestionBase):
    """A question answered by picking one of its named options."""

    type: Literal["choice"] = "choice"
    criteria: dict[str, Guidance | None] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)


class ScoreQuestion(_QuestionBase):
    """A question answered on a graded scale of guidance levels."""

    type: Literal["score"] = "score"
    criteria: list[Guidance] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class JevRequest(BaseModel):
    """One state evaluated against one or more questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    state: str | dict[str, JsonValue] | list[JsonValue]
    questions: dict[str, Question] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    """Probability of a yes answer; noul answers carry no confidence."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["noul"] = "noul"
    noul: float

    @field_validator("noul")
    @classmethod
    def _normalise(cls, value: float) -> float:
        return normalise_noul(value)


class ChoiceAnswer(BaseModel):
    """Jev's answer to a choice question: the pick and, when served, its probabilities."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None

    @field_validator("probabilities")
    @classmethod
    def _normalise(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        return None if value is None else normalise_choice_probabilities(value)


class ScoreAnswer(BaseModel):
    """``score`` is the expected 0-based level index; keys are level indices as strings."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, JsonValue]
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class JevUsage(BaseModel):
    """Token counts and cost Jev reported for one request."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    input_tokens: int
    output_tokens: int
    cost: float | None = None


class JevResponse(BaseModel):
    """A parsed reply. ``id`` is the OpenRouter generation id when present."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    answers: dict[str, Answer]
    usage: JevUsage
    id: str | None = None


def parse_response(payload: Mapping[str, Any], requested_ids: Iterable[str]) -> JevResponse:
    """Validate and normalise a reply; raise ``IncompleteAnswersError`` on a missing id.

    Pydantic validation errors propagate unchanged; callers map them to a
    ``DecisionRequestError`` without attaching the body.
    """
    response = JevResponse.model_validate(payload)
    missing = sorted(set(requested_ids) - set(response.answers))
    if missing:
        raise IncompleteAnswersError(f"reply omitted answers for: {', '.join(missing)}")
    return response


# ---------------------------------------------------------------------------
# Call records
# ---------------------------------------------------------------------------


_COST_SOURCES: frozenset[str] = frozenset({"billed", "computed"})


@dataclass(frozen=True, slots=True)
class JevCall:
    """One completed request.

    ``billed_cost_usd`` is the OpenRouter ``usage.cost`` (``None`` on the
    direct route); ``cost_usd`` equals it when present (``cost_source =
    'billed'``), else the cost computed from the configured input price
    (``'computed'``). ``list_cost_usd`` is always the computed list price.
    """

    input_tokens: int
    output_tokens: int
    billed_cost_usd: float | None
    cost_usd: float
    cost_source: CostSource
    list_cost_usd: float
    latency_s: float
    served_model: str | None
    request_id: str | None
    attempts: int

    def __post_init__(self) -> None:
        """Reject a cost source other than ``billed`` or ``computed``."""
        if self.cost_source not in _COST_SOURCES:
            raise ValueError(
                f"cost_source must be 'billed' or 'computed', not {self.cost_source!r}"
            )


@dataclass(frozen=True, slots=True)
class JevAttempt:
    """One HTTP attempt; ``status`` is ``None`` when no response arrived. Holds no body."""

    chunk_index: int
    attempt: int
    status: int | None
    error_type: str | None
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    billed_cost_usd: float | None
    served_model: str | None
    request_id: str | None


@dataclass(slots=True)
class JevResult:
    """Normalised answers for every requested id plus one ``JevCall`` per request."""

    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    requested_model: str
    served_model: str | None
    route: Route
    calls: list[JevCall] = field(default_factory=list)
