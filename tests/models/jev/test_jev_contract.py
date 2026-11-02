"""Contract tests: documented reply shapes run through ``JevClient`` offline.

Every reply comes from a static, hand-written fixture under ``fixtures/``:
``direct/`` holds bodies written from the direct route's documented response
and error shapes, ``openrouter/`` bodies written from OpenRouter's documented
examples. No reply was recorded from a live service; every file says so with
``"synthetic": true``. Each fixture is served through the scripted fake
transport, so the whole client path (send, map, retry, parse) runs.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.models.jev._jev_fake_transport import SYNTHETIC_REQUEST_ID_PREFIX, FakeJev, Reply

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV, JevSettings
from osprey.models.jev.errors import (
    REQUEST_ID_HEADER,
    DecisionAuthError,
    DecisionError,
    DecisionRequestError,
    DecisionUnavailableError,
)
from osprey.models.jev.models import ChoiceAnswer, NoulAnswer, ScoreAnswer

FIXTURES = Path(__file__).parent / "fixtures"

SYNTHETIC_KEY = "synthetic-key-not-real"
TYPESAFE_BASE = "https://synthetic-typesafe.invalid"
OPENROUTER_BASE = "https://synthetic-openrouter.invalid/api/v1"
# The 422 fixture and the OpenRouter 502 fixture echo exactly this state.
STATE = "synthetic state: find the warm widget in bay seven"

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


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _config(route: str) -> dict[str, Any]:
    if route == "typesafe":
        return {
            "api": {
                "jev": {
                    "routes": {"typesafe": {"base_url": TYPESAFE_BASE, "api_key": SYNTHETIC_KEY}}
                }
            }
        }
    return {
        "api": {
            "jev": {"route": "openrouter"},
            "providers": {"openrouter": {"base_url": OPENROUTER_BASE, "api_key": SYNTHETIC_KEY}},
        }
    }


class _OpenRouterWire(httpx.BaseTransport):
    """The fake transport without the direct route's request-id header.

    OpenRouter identifies a reply by the body ``id`` only, so the header the
    fake stamps on every reply is removed.
    """

    def __init__(self, fake: FakeJev) -> None:
        self.fake = fake

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.fake.transport.handle_request(request)
        del response.headers[REQUEST_ID_HEADER]
        return response


def _client(fake: FakeJev, route: str) -> JevClient:
    config = _config(route)
    transport: httpx.BaseTransport = (
        fake.transport if route == "typesafe" else _OpenRouterWire(fake)
    )
    return JevClient(
        JevSettings.from_config(config),
        config,
        transport=transport,
        clock=fake.clock,
        sleep=fake.sleep,
    )


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _reply(name: str) -> Reply:
    doc = _load(name)
    return Reply(status=doc["status"], body=doc["body"], headers=doc["headers"])


def _route_of(name: str) -> str:
    return "typesafe" if name.startswith("direct/") else "openrouter"


def _decide_error(
    fake: FakeJev, route: str, questions: dict[str, Any] | None = None
) -> DecisionError:
    with _client(fake, route) as client:
        with pytest.raises(DecisionError) as info:
            client.decide(STATE, questions or {"q1": NOUL}, budget="interactive")
    return info.value


def _assert_no_chain(exc: BaseException) -> None:
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert STATE not in "".join(traceback.format_exception(exc))


# ---------------------------------------------------------------------------
# The fixtures themselves
# ---------------------------------------------------------------------------

ALL_FIXTURES = sorted(
    str(path.relative_to(FIXTURES).with_suffix("")) for path in FIXTURES.rglob("*.json")
)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_fixture_is_labelled_synthetic(name: str) -> None:
    doc = _load(name)
    assert doc["synthetic"] is True
    assert doc["route"] == _route_of(name)
    assert isinstance(doc["status"], int)
    assert isinstance(doc["headers"], dict)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_fixtures_carry_no_session_references(name: str) -> None:
    text = (FIXTURES / f"{name}.json").read_text()
    assert ".claude" + "/scratch" not in text
    assert "M\u00a7" not in text
    assert not re.search(r"(?<![A-Z0-9_])E:\d", text)


def test_the_state_echo_fixtures_echo_the_state_sent() -> None:
    assert STATE in json.dumps(_load("direct/422_detail_list"))
    assert STATE in json.dumps(_load("openrouter/200_with_error_502"))


# ---------------------------------------------------------------------------
# Successful replies
# ---------------------------------------------------------------------------


def test_direct_answers_of_every_type_parse() -> None:
    fake = FakeJev(responses=[_reply("direct/200_answers")])
    with _client(fake, "typesafe") as client:
        result = client.decide(
            STATE, {"q_noul": NOUL, "q_choice": CHOICE, "q_score": SCORE}, budget="interactive"
        )

    assert isinstance(result.answers["q_noul"], NoulAnswer)
    assert result.answers["q_noul"].noul == pytest.approx(0.87)
    choice = result.answers["q_choice"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice == "o1"
    assert choice.probabilities == {"o1": 0.81, "o2": 0.19}
    score = result.answers["q_score"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == pytest.approx(1.6)
    assert score.legend["2"] == "hot"

    (call,) = result.calls
    assert call.request_id == f"{SYNTHETIC_REQUEST_ID_PREFIX}1"
    assert call.billed_cost_usd is None
    assert call.cost_source == "computed"
    assert call.input_tokens == 412
    assert result.served_model == "jev-1.13.0"


def test_openrouter_answers_carry_the_billed_cost_and_body_id() -> None:
    fake = FakeJev(responses=[_reply("openrouter/200_answers_with_cost")])
    with _client(fake, "openrouter") as client:
        result = client.decide(STATE, {"q_noul": NOUL, "q_choice": CHOICE}, budget="interactive")

    assert isinstance(result.answers["q_noul"], NoulAnswer)
    assert isinstance(result.answers["q_choice"], ChoiceAnswer)
    (call,) = result.calls
    assert call.request_id == "gen-dec-synthetic-0001"
    assert call.billed_cost_usd == pytest.approx(0.0000163)
    assert call.cost_usd == pytest.approx(0.0000163)
    assert call.cost_source == "billed"
    assert result.route == "openrouter"


# ---------------------------------------------------------------------------
# Error replies: status, class and the error_type table
# ---------------------------------------------------------------------------

# (fixture, class, status, error_type, calls made on the interactive budget)
ERROR_CASES = [
    ("direct/200_answer_missing_type", DecisionRequestError, 200, "malformed_response", 1),
    ("direct/400_detail_object", DecisionRequestError, 400, "max_tokens_exceeded", 1),
    ("direct/400_detail_string", DecisionRequestError, 400, "api_usage_error", 1),
    ("direct/422_detail_list", DecisionRequestError, 422, "validation_error", 1),
    ("direct/401_unauthorized", DecisionAuthError, 401, "api_usage_error", 1),
    ("direct/402_payment_required", DecisionUnavailableError, 402, "insufficient_balance", 1),
    ("direct/413_payload_too_large", DecisionRequestError, 413, "max_tokens_exceeded", 1),
    ("direct/429_rate_limited", DecisionUnavailableError, 429, "api_usage_error", 1),
    ("direct/500_server_error", DecisionUnavailableError, 500, "api_usage_error", 2),
    ("direct/503_no_body", DecisionUnavailableError, 503, "http_503", 2),
    ("direct/524_html_timeout", DecisionUnavailableError, 524, "http_524", 2),
    ("openrouter/200_with_error_502", DecisionUnavailableError, 502, "openrouter_502", 1),
    ("openrouter/200_with_error_429", DecisionUnavailableError, 429, "openrouter_429", 1),
    ("openrouter/401_invalid_key", DecisionAuthError, 401, "openrouter_401", 1),
    ("openrouter/402_insufficient_credits", DecisionUnavailableError, 402, "openrouter_402", 1),
    ("openrouter/429_rate_limited", DecisionUnavailableError, 429, "openrouter_429", 1),
    ("openrouter/502_bad_gateway", DecisionUnavailableError, 502, "openrouter_502", 2),
]


@pytest.mark.parametrize(
    ("name", "cls", "status", "error_type", "calls"),
    ERROR_CASES,
    ids=[case[0] for case in ERROR_CASES],
)
def test_error_reply_maps_to_its_tabled_decision_error(
    name: str, cls: type[DecisionError], status: int, error_type: str, calls: int
) -> None:
    # Every attempt, first and retry, gets the same reply.
    fake = FakeJev(responses=[_reply(name)] * 3)
    exc = _decide_error(fake, _route_of(name))

    assert type(exc) is cls
    assert exc.status == status
    assert exc.error_type == error_type
    assert fake.call_count == calls
    if _route_of(name) == "typesafe":
        assert exc.request_id == f"{SYNTHETIC_REQUEST_ID_PREFIX}{calls}"
    else:
        assert exc.request_id == _load(name)["body"]["id"]
    _assert_no_chain(exc)


def test_every_fixture_is_exercised() -> None:
    used = {case[0] for case in ERROR_CASES} | {
        "direct/200_answers",
        "openrouter/200_answers_with_cost",
    }
    assert used == set(ALL_FIXTURES)


def test_openrouter_200_with_error_carries_the_error_code_as_status() -> None:
    fake = FakeJev(responses=[_reply("openrouter/200_with_error_502")])
    exc = _decide_error(fake, "openrouter")
    assert exc.status == _load("openrouter/200_with_error_502")["body"]["error"]["code"]
    assert "Provider returned error" in str(exc)
    assert STATE not in str(exc)


def test_html_524_keeps_the_page_out_of_the_message() -> None:
    fake = FakeJev(responses=[_reply("direct/524_html_timeout")] * 3)
    exc = _decide_error(fake, "typesafe")
    assert "<html" not in str(exc)
    assert "timed out responding" not in str(exc)


# ---------------------------------------------------------------------------
# Failures without a response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "error_type"),
    [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "transport")],
    ids=["read-timeout", "connect-error"],
)
def test_responseless_failure_has_no_status(
    raised: type[httpx.RequestError], error_type: str
) -> None:
    fake = FakeJev(responses=[raised] * 3)
    exc = _decide_error(fake, "typesafe")

    assert type(exc) is DecisionUnavailableError
    assert exc.status is None
    assert exc.error_type == error_type
    assert exc.request_id is None
    # Interactive: one retry on a timeout or connection failure.
    assert fake.call_count == 2
    _assert_no_chain(exc)


# ---------------------------------------------------------------------------
# A 422 echoing the state leaks it nowhere
# ---------------------------------------------------------------------------


def test_422_state_echo_stays_out_of_message_logs_and_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    fake = FakeJev(responses=[_reply("direct/422_detail_list")])
    exc = _decide_error(fake, "typesafe")

    assert type(exc) is DecisionRequestError
    assert exc.error_type == "validation_error"
    assert "at least 1 character" in str(exc)
    assert STATE not in str(exc)
    assert STATE not in repr(exc)
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert STATE not in "".join(traceback.format_exception(exc))
    for record in caplog.records:
        assert STATE not in record.getMessage()
        assert STATE not in (record.exc_text or "")
