"""The single-request path of ``JevClient``: body, endpoint, cost, parsing and errors.

Every test runs offline on the scripted fake transport. All data is synthetic.
"""

from __future__ import annotations

import traceback

import httpx
import pytest
from pydantic import ValidationError
from tests.models.jev._jev_fake_transport import SYNTHETIC_REQUEST_ID_PREFIX, FakeJev, Reply

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV, JevSettings
from osprey.models.jev.errors import (
    DecisionAuthError,
    DecisionError,
    DecisionRequestError,
    DecisionUnavailableError,
    IncompleteAnswersError,
)
from osprey.models.jev.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevAttempt,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
)

SYNTHETIC_KEY = "synthetic-key-not-real"
TYPESAFE_BASE = "https://synthetic-typesafe.invalid"
OPENROUTER_BASE = "https://synthetic-openrouter.invalid/api/v1"
STATE = "synthetic state: find the warm widget"
PRICE = 0.042

NOUL = {"type": "noul", "instructions": "Is the synthetic widget warm?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which synthetic kind?",
    "criteria": {"o1": "alpha", "o2": "beta"},
}
SCORE = {
    "type": "score",
    "instructions": "How warm is the synthetic widget?",
    "criteria": ["cold", "tepid", "hot"],
}


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (TYPESAFE_KEY_ENV, OPENROUTER_KEY_ENV, "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _typesafe_config(**jev: object) -> dict:
    return {
        "api": {
            "jev": {
                "routes": {"typesafe": {"base_url": TYPESAFE_BASE, "api_key": SYNTHETIC_KEY}},
                **jev,
            }
        }
    }


def _openrouter_config() -> dict:
    return {
        "api": {
            "jev": {"route": "openrouter"},
            "providers": {"openrouter": {"base_url": OPENROUTER_BASE, "api_key": SYNTHETIC_KEY}},
        }
    }


class _Recorder(httpx.BaseTransport):
    """Delegates to the fake and keeps every ``httpx.Request`` for header/URL checks."""

    def __init__(self, fake: FakeJev) -> None:
        self.fake = fake
        self.seen: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return self.fake.transport.handle_request(request)


def _client(fake: FakeJev, config: dict | None = None, **kwargs: object) -> JevClient:
    config = config if config is not None else _typesafe_config()
    return JevClient(
        JevSettings.from_config(config),
        config,
        transport=fake.transport,
        clock=fake.clock,
        sleep=fake.sleep,
        **kwargs,
    )


def _no_chained_body(exc: BaseException, *needles: str) -> None:
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert not isinstance(exc.__context__, ValidationError)
    text = "".join(traceback.format_exception(exc))
    assert "ValidationError" not in text
    assert "input_value" not in text
    for needle in needles:
        assert needle not in text


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_score_decide_round_trips_on_the_fake_transport() -> None:
    fake = FakeJev(
        {
            "q1": {
                "type": "score",
                "score": 1.73,
                "legend": {"0": "cold", "1": "tepid", "2": "hot"},
                "probabilities": {"0": 0.05, "1": 0.17, "2": 0.78},
                "confidence": 0.8,
            }
        },
        model="jev-1.13.0-20260917",
    )
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": SCORE}, budget="batch")

    answer = result.answers["q1"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(1.73)
    assert answer.legend["2"] == "hot"
    assert result.requested_model == "jev-1.13.0"
    assert result.served_model == "jev-1.13.0-20260917"
    assert result.route == "typesafe"
    assert len(result.calls) == 1
    assert fake.call_count == 1

    sent = fake.requests[0]
    assert sent["model"] == "jev-1.13.0"
    assert sent["state"] == STATE
    assert sent["questions"]["q1"]["type"] == "score"
    assert sent["questions"]["q1"]["criteria"] == ["cold", "tepid", "hot"]


def test_request_goes_to_the_resolved_endpoint_with_a_bearer_key() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.9}})
    recorder = _Recorder(fake)
    config = _typesafe_config()
    with JevClient(
        JevSettings.from_config(config), config, transport=recorder, clock=fake.clock
    ) as client:
        client.decide(STATE, {"q1": NOUL}, budget="interactive")

    (request,) = recorder.seen
    assert request.method == "POST"
    assert str(request.url) == f"{TYPESAFE_BASE}/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {SYNTHETIC_KEY}"


def test_model_is_always_sent_and_typed_questions_are_accepted() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.9}, "q2": {"type": "choice", "choice": "o1"}})
    questions = {
        "q1": NoulQuestion(instructions="Is the synthetic widget warm?"),
        "q2": ChoiceQuestion(
            instructions="Which synthetic kind?", criteria={"o1": "a", "o2": None}
        ),
    }
    with _client(fake, _openrouter_config()) as client:
        client.decide({"synthetic": "state"}, questions, budget="batch")

    sent = fake.requests[0]
    assert sent["model"] == "typesafe/jev-1.13"
    assert sent["state"] == {"synthetic": "state"}
    assert sent["questions"]["q1"] == {
        "type": "noul",
        "instructions": "Is the synthetic widget warm?",
    }
    assert sent["questions"]["q2"]["criteria"] == {"o1": "a", "o2": None}


def test_answers_are_normalised_at_parse() -> None:
    fake = FakeJev(
        {
            "q1": {"type": "noul", "noul": 0.91234},
            "q2": {
                "type": "choice",
                "choice": "o1",
                "probabilities": {"o1": 0.5700000000000001, "o2": 0.43, "o3": 0.001},
            },
        }
    )
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": NOUL, "q2": CHOICE}, budget="batch")

    assert isinstance(result.answers["q1"], NoulAnswer)
    assert result.answers["q1"].noul == pytest.approx(0.912)
    assert result.answers["q2"].probabilities == {"o1": 0.57, "o2": 0.43}


def test_choice_without_probabilities_parses() -> None:
    fake = FakeJev({"q1": {"type": "choice", "choice": "o2"}})
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": CHOICE}, budget="batch")

    answer = result.answers["q1"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "o2"
    assert answer.probabilities is None
    assert answer.confidence is None


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


def test_cost_source_billed_from_openrouter_usage_cost() -> None:
    fake = FakeJev(
        {"q1": {"type": "noul", "noul": 0.5}},
        usage={"input_tokens": 1_000, "output_tokens": 5, "cost": 0.00321},
    )
    with _client(fake, _openrouter_config()) as client:
        result = client.decide(STATE, {"q1": NOUL}, budget="batch")

    (call,) = result.calls
    assert result.route == "openrouter"
    assert call.cost_source == "billed"
    assert call.billed_cost_usd == pytest.approx(0.00321)
    assert call.cost_usd == pytest.approx(0.00321)
    assert call.list_cost_usd == pytest.approx(1_000 * PRICE / 1_000_000)
    assert call.input_tokens == 1_000
    assert call.output_tokens == 5
    assert call.attempts == 1
    assert call.request_id == f"{SYNTHETIC_REQUEST_ID_PREFIX}1"


def test_cost_source_computed_from_the_configured_input_price() -> None:
    fake = FakeJev(
        {"q1": {"type": "noul", "noul": 0.5}},
        usage={"input_tokens": 2_000, "output_tokens": 5},
    )
    config = _typesafe_config(input_cost_per_million_tokens=1.5)
    with _client(fake, config) as client:
        result = client.decide(STATE, {"q1": NOUL}, budget="batch")

    (call,) = result.calls
    assert call.cost_source == "computed"
    assert call.billed_cost_usd is None
    assert call.cost_usd == pytest.approx(2_000 * 1.5 / 1_000_000)
    assert call.list_cost_usd == call.cost_usd


def test_openrouter_without_usage_cost_is_computed() -> None:
    fake = FakeJev(
        {"q1": {"type": "noul", "noul": 0.5}},
        usage={"input_tokens": 1_000, "output_tokens": 5},
    )
    with _client(fake, _openrouter_config()) as client:
        (call,) = client.decide(STATE, {"q1": NOUL}, budget="batch").calls

    assert call.cost_source == "computed"
    assert call.billed_cost_usd is None
    assert call.cost_usd == pytest.approx(1_000 * PRICE / 1_000_000)


def test_openrouter_request_id_falls_back_to_the_body_id() -> None:
    body = {
        "id": "gen-dec-synthetic-1",
        "model": "typesafe/jev-1.13",
        "answers": {"q1": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.0001},
    }
    fake = FakeJev(responses=[Reply(status=200, body=body)])
    recorder_fake = fake

    class _StripHeader(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            response = recorder_fake.transport.handle_request(request)
            headers = {k: v for k, v in response.headers.items() if k != "x-typesafe-request-id"}
            return httpx.Response(response.status_code, headers=headers, content=response.read())

    config = _openrouter_config()
    with JevClient(
        JevSettings.from_config(config), config, transport=_StripHeader(), clock=fake.clock
    ) as client:
        (call,) = client.decide(STATE, {"q1": NOUL}, budget="batch").calls
    assert call.request_id == "gen-dec-synthetic-1"


# ---------------------------------------------------------------------------
# Keys and local refusals: no HTTP
# ---------------------------------------------------------------------------


def test_construct_without_a_key_never_raises_and_decide_raises_no_key() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    config = {"api": {"jev": {"routes": {"typesafe": {"base_url": TYPESAFE_BASE}}}}}
    client = _client(fake, config)  # must not raise

    with pytest.raises(DecisionAuthError) as info:
        client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert info.value.error_type == "no_key"
    assert info.value.status is None
    assert fake.requests == []
    assert fake.call_count == 0
    client.close()


def test_a_key_set_after_construction_is_picked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    config = {"api": {"jev": {"routes": {"typesafe": {"base_url": TYPESAFE_BASE}}}}}
    with _client(fake, config) as client:
        with pytest.raises(DecisionAuthError):
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
        monkeypatch.setenv(TYPESAFE_KEY_ENV, SYNTHETIC_KEY)
        client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert fake.call_count == 1


@pytest.mark.parametrize(
    "questions",
    [
        pytest.param({"q1": {"type": "noul", "instructions": "   "}}, id="blank-instructions"),
        pytest.param({"q1": {"type": "noul"}}, id="missing-instructions"),
        pytest.param(
            {"q1": {"type": "noul", "instructions": "x", "criteria": {"true": "yes"}}},
            id="noul-criteria-one-side",
        ),
        pytest.param(
            {"q1": {"type": "score", "instructions": "x", "criteria": ["only"]}},
            id="score-one-level",
        ),
        pytest.param(
            {"q1": {"type": "choice", "instructions": "x", "criteria": {}}}, id="choice-empty"
        ),
        pytest.param({"q1": {"type": "mystery", "instructions": "x"}}, id="unknown-type"),
        pytest.param({}, id="no-questions"),
    ],
)
def test_invalid_request_is_refused_before_any_http(questions: dict) -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, questions, budget="batch")

    assert info.value.error_type == "invalid_request"
    assert info.value.status is None
    assert fake.call_count == 0
    _no_chained_body(info.value, STATE)


def test_choice_over_the_configured_option_cap_is_refused_before_any_http() -> None:
    fake = FakeJev({"q1": {"type": "choice", "choice": "o1"}})
    config = _typesafe_config(max_choice_options=2)
    question = {
        "type": "choice",
        "instructions": "Which synthetic kind?",
        "criteria": {"o1": "a", "o2": "b", "o3": "c"},
    }
    with _client(fake, config) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, {"q1": question}, budget="batch")
    assert info.value.error_type == "invalid_request"
    assert fake.call_count == 0


def test_unknown_budget_is_refused_before_any_http() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="leisurely")
    assert info.value.error_type == "invalid_request"
    assert fake.call_count == 0


def test_a_passed_deadline_raises_without_http() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch", deadline=fake.clock() - 1.0)
    assert info.value.error_type == "deadline_exceeded"
    assert info.value.status is None
    assert fake.call_count == 0


def test_new_deadline_adds_the_budget_total_to_the_clock() -> None:
    fake = FakeJev()
    with _client(fake) as client:
        assert client.new_deadline("interactive") == pytest.approx(1003.0)
        assert client.new_deadline("batch") == pytest.approx(1030.0)


# ---------------------------------------------------------------------------
# Reply validation and error mapping
# ---------------------------------------------------------------------------


def test_missing_answer_id_raises_incomplete_with_the_billed_call() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    with _client(fake) as client:
        with pytest.raises(IncompleteAnswersError) as info:
            client.decide(STATE, {"q1": NOUL, "q2": NOUL}, budget="batch")

    exc = info.value
    assert "q2" in exc.message
    assert exc.error_type == "incomplete_answers"
    assert exc.status == 200
    assert exc.request_id == f"{SYNTHETIC_REQUEST_ID_PREFIX}1"
    assert len(exc.completed_calls) == 1
    _no_chained_body(exc)


def test_answer_missing_type_is_malformed_and_never_chained() -> None:
    body = {
        "model": "jev-1.13.0",
        "answers": {"q1": {"noul": 0.5}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    fake = FakeJev(responses=[Reply(status=200, body=body)])
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.error_type == "malformed_response"
    assert info.value.status == 200
    _no_chained_body(info.value)


def test_undecodable_200_is_malformed() -> None:
    fake = FakeJev(responses=[Reply(status=200, body=b"<html>not json</html>")])
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.error_type == "malformed_response"
    _no_chained_body(info.value)


def test_openrouter_200_with_an_error_object_maps_by_code() -> None:
    body = {"id": "gen-dec-synthetic-2", "error": {"code": 402, "message": "synthetic: no credit"}}
    fake = FakeJev(responses=[Reply(status=200, body=body)])
    with _client(fake, _openrouter_config()) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.status == 402
    assert info.value.error_type == "openrouter_402"


def test_401_is_an_auth_error() -> None:
    fake = FakeJev(responses=[Reply(status=401, body={"detail": "synthetic: bad key"})])
    with _client(fake) as client:
        with pytest.raises(DecisionAuthError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert info.value.status == 401
    assert info.value.request_id == f"{SYNTHETIC_REQUEST_ID_PREFIX}1"


def test_422_echoing_the_state_leaves_it_out_of_the_error() -> None:
    body = {
        "detail": [
            {"loc": ["body", "questions"], "msg": "synthetic invalid", "input": {"state": STATE}}
        ]
    }
    fake = FakeJev(responses=[Reply(status=422, body=body)])
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.error_type == "validation_error"
    _no_chained_body(info.value, STATE)


def test_a_timeout_is_unavailable_with_no_status() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}}, latency=5.0)
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert info.value.error_type == "timeout"
    assert info.value.status is None
    _no_chained_body(info.value)


def test_a_connection_error_is_a_transport_failure() -> None:
    # One ConnectError per attempt the batch budget allows (1 + max_retries=2).
    fake = FakeJev(responses=[httpx.ConnectError] * 3)
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.error_type == "transport"
    assert info.value.status is None
    _no_chained_body(info.value)


def test_every_failure_is_a_decision_error() -> None:
    fake = FakeJev(responses=[503])
    with _client(fake) as client:
        with pytest.raises(DecisionError):
            client.decide(STATE, {"q1": NOUL}, budget="interactive")


# ---------------------------------------------------------------------------
# Attempt records
# ---------------------------------------------------------------------------


def test_on_attempt_fires_once_per_attempt_without_a_body() -> None:
    attempts: list[JevAttempt] = []
    fake = FakeJev(
        {"q1": {"type": "noul", "noul": 0.5}},
        usage={"input_tokens": 100, "output_tokens": 5, "cost": 0.001},
        latency=0.25,
    )
    with _client(fake, _openrouter_config(), on_attempt=attempts.append) as client:
        client.decide(STATE, {"q1": NOUL}, budget="batch")

    (attempt,) = attempts
    assert attempt.chunk_index == 0
    assert attempt.attempt == 1
    assert attempt.status == 200
    assert attempt.error_type is None
    assert attempt.latency_s == pytest.approx(0.25)
    assert attempt.input_tokens == 100
    assert attempt.billed_cost_usd == pytest.approx(0.001)
    assert attempt.request_id == f"{SYNTHETIC_REQUEST_ID_PREFIX}1"


def test_on_attempt_records_a_failed_attempt() -> None:
    attempts: list[JevAttempt] = []
    fake = FakeJev(responses=[Reply(status=401)])
    with _client(fake, on_attempt=attempts.append) as client:
        with pytest.raises(DecisionAuthError):
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
    (attempt,) = attempts
    assert attempt.status == 401
    assert attempt.error_type == "http_401"
    assert attempt.input_tokens is None


def test_latency_is_measured_on_the_injected_clock() -> None:
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}}, latency=1.5)
    with _client(fake) as client:
        (call,) = client.decide(STATE, {"q1": NOUL}, budget="batch").calls
    assert call.latency_s == pytest.approx(1.5)
