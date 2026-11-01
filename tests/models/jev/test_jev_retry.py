"""The retry budgets of ``JevClient``: per-call timeouts, retries, waits and the deadline.

Every test runs offline on the scripted fake transport and its fake clock; the
injected ``sleep`` only advances that clock. All data is synthetic.
"""

from __future__ import annotations

import time

import httpx
import pytest
from tests.models.jev._jev_fake_transport import FakeJev, Reply

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV, JevSettings
from osprey.models.jev.errors import (
    DecisionRequestError,
    DecisionUnavailableError,
)
from osprey.models.jev.models import JevAttempt

SYNTHETIC_KEY = "synthetic-key-not-real"
TYPESAFE_BASE = "https://synthetic-typesafe.invalid"
STATE = "synthetic state: find the warm widget"
NOUL = {"type": "noul", "instructions": "Is the synthetic widget warm?"}
ANSWERS = {"q1": {"type": "noul", "noul": 0.5}}


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (TYPESAFE_KEY_ENV, OPENROUTER_KEY_ENV, "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _config() -> dict:
    return {
        "api": {
            "jev": {"routes": {"typesafe": {"base_url": TYPESAFE_BASE, "api_key": SYNTHETIC_KEY}}}
        }
    }


class _TimeoutRecorder(httpx.BaseTransport):
    """Delegates to the fake and keeps the read timeout each attempt was sent with."""

    def __init__(self, fake: FakeJev) -> None:
        self.fake = fake
        self.timeouts: list[float | None] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.timeouts.append((request.extensions.get("timeout") or {}).get("read"))
        return self.fake.transport.handle_request(request)


def _client(
    fake: FakeJev, *, transport: httpx.BaseTransport | None = None, **kwargs: object
) -> JevClient:
    config = _config()
    return JevClient(
        JevSettings.from_config(config),
        config,
        transport=transport if transport is not None else fake.transport,
        clock=fake.clock,
        sleep=fake.sleep,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 429: interactive raises at once, batch retries
# ---------------------------------------------------------------------------


def test_interactive_429_raises_at_once() -> None:
    fake = FakeJev(ANSWERS, responses=[429])
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert info.value.status == 429
    assert fake.call_count == 1
    assert fake.clock() == pytest.approx(1000.0)


def test_batch_429_then_200_succeeds() -> None:
    fake = FakeJev(ANSWERS, responses=[429])
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert fake.call_count == 2
    (call,) = result.calls
    assert call.attempts == 2
    assert result.answers["q1"].noul == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Retryable failures and retry counts
# ---------------------------------------------------------------------------


def test_interactive_retries_a_5xx_once() -> None:
    fake = FakeJev(ANSWERS, responses=[503])
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert fake.call_count == 2
    assert result.calls[0].attempts == 2


def test_interactive_gives_up_after_one_retry() -> None:
    fake = FakeJev(ANSWERS, responses=[503, 503, None])
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert info.value.status == 503
    assert fake.call_count == 2


def test_interactive_retries_a_connection_error() -> None:
    fake = FakeJev(ANSWERS, responses=[httpx.ConnectError])
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert fake.call_count == 2
    assert result.calls[0].attempts == 2


def test_batch_retries_twice_then_raises_the_last_error() -> None:
    fake = FakeJev(ANSWERS, responses=[503, 502, 500, None])
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.status == 500
    assert fake.call_count == 3


@pytest.mark.parametrize("budget", ["interactive", "batch"])
def test_a_request_error_is_never_retried(budget: str) -> None:
    fake = FakeJev(ANSWERS, responses=[400])
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError):
            client.decide(STATE, {"q1": NOUL}, budget=budget)
    assert fake.call_count == 1


def test_an_incomplete_reply_is_never_retried() -> None:
    fake = FakeJev({})  # every reply omits q1
    with _client(fake) as client:
        with pytest.raises(DecisionRequestError):
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# The deadline and per-attempt timeouts
# ---------------------------------------------------------------------------


def test_new_deadline_is_total_seconds_on_the_injected_clock() -> None:
    fake = FakeJev(ANSWERS)
    with _client(fake) as client:
        assert client.new_deadline("interactive") == pytest.approx(1003.0)
        assert client.new_deadline("batch") == pytest.approx(1030.0)


def test_a_decide_given_a_deadline_raises_after_three_fake_seconds() -> None:
    fake = FakeJev(ANSWERS, latency=10.0)
    with _client(fake) as client:
        deadline = client.new_deadline("interactive")
        assert deadline == pytest.approx(1003.0)
        with pytest.raises(DecisionUnavailableError):
            client.decide(STATE, {"q1": NOUL}, budget="interactive", deadline=deadline)
    assert fake.clock() == pytest.approx(1003.0)


def test_a_slow_chain_raises_inside_the_interactive_budget() -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=503, latency=1.8)], latency=1.8)
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert info.value.error_type == "timeout"
    assert fake.clock() <= 1003.0 + 1e-9
    assert fake.call_count == 2


def test_attempt_timeout_is_the_remaining_deadline_when_shorter() -> None:
    fake = FakeJev(ANSWERS)
    recorder = _TimeoutRecorder(fake)
    with _client(fake, transport=recorder) as client:
        client.decide(STATE, {"q1": NOUL}, budget="interactive", deadline=fake.clock() + 1.2)
    assert recorder.timeouts == [pytest.approx(1.2)]


def test_a_retry_timeout_is_clipped_to_what_is_left() -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=503, latency=1.0)])
    recorder = _TimeoutRecorder(fake)
    with _client(fake, transport=recorder) as client:
        client.decide(STATE, {"q1": NOUL}, budget="interactive")
    first, second = recorder.timeouts
    assert first == pytest.approx(2.0)
    assert second is not None and second < 2.0
    assert second == pytest.approx(1003.0 - (fake.clock()))  # the fake answers instantly


def test_the_attempt_timeout_is_the_budget_timeout_when_shorter() -> None:
    fake = FakeJev(ANSWERS)
    recorder = _TimeoutRecorder(fake)
    with _client(fake, transport=recorder) as client:
        client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert recorder.timeouts == [pytest.approx(10.0)]


def test_a_passed_deadline_raises_before_any_http() -> None:
    fake = FakeJev(ANSWERS)
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch", deadline=fake.clock() - 1.0)
    assert info.value.error_type == "deadline_exceeded"
    assert fake.call_count == 0


# ---------------------------------------------------------------------------
# Waits: retry-after, retry-after-ms, the interactive cap, exponential backoff
# ---------------------------------------------------------------------------


def test_batch_honours_retry_after_on_the_fake_clock_only() -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=503, headers={"retry-after": "20"})])
    started = time.monotonic()
    with _client(fake) as client:
        result = client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert time.monotonic() - started < 0.5
    assert fake.clock() == pytest.approx(1020.0)
    assert result.calls[0].attempts == 2


def test_batch_honours_retry_after_ms() -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=429, headers={"retry-after-ms": "1500"})])
    with _client(fake) as client:
        client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert fake.clock() == pytest.approx(1001.5)


def test_interactive_caps_the_wait_at_one_second() -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=503, headers={"retry-after": "20"})])
    with _client(fake) as client:
        client.decide(STATE, {"q1": NOUL}, budget="interactive")
    assert fake.clock() == pytest.approx(1001.0)


def test_a_wait_past_the_deadline_stops_without_sleeping() -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=503, headers={"retry-after": "40"})])
    with _client(fake) as client:
        with pytest.raises(DecisionUnavailableError) as info:
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert info.value.status == 503
    assert fake.call_count == 1
    assert fake.clock() == pytest.approx(1000.0)


def test_without_retry_after_the_wait_grows_exponentially() -> None:
    waits: list[float] = []
    fake = FakeJev(ANSWERS, responses=[503, 503])

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        fake.sleep(seconds)

    config = _config()
    with JevClient(
        JevSettings.from_config(config),
        config,
        transport=fake.transport,
        clock=fake.clock,
        sleep=sleep,
    ) as client:
        client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert len(waits) == 2
    assert waits[0] > 0
    assert waits[1] == pytest.approx(2 * waits[0])


@pytest.mark.parametrize("value", ["soon", "-3", "Wed, 21 Oct 2015 07:28:00 GMT"])
def test_an_unusable_retry_after_falls_back_to_backoff(value: str) -> None:
    fake = FakeJev(ANSWERS, responses=[Reply(status=503, headers={"retry-after": value})])
    with _client(fake) as client:
        client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert 1000.0 < fake.clock() < 1002.0


# ---------------------------------------------------------------------------
# Attempt records
# ---------------------------------------------------------------------------


def test_503_then_200_reports_two_attempts() -> None:
    attempts: list[JevAttempt] = []
    fake = FakeJev(ANSWERS, responses=[503])
    with _client(fake, on_attempt=attempts.append) as client:
        client.decide(STATE, {"q1": NOUL}, budget="batch")
    first, second = attempts
    assert (first.attempt, first.status, first.error_type) == (1, 503, "http_503")
    assert (second.attempt, second.status, second.error_type) == (2, 200, None)
    assert first.input_tokens is None
    assert second.input_tokens is not None


def test_every_failed_attempt_is_reported() -> None:
    attempts: list[JevAttempt] = []
    fake = FakeJev(ANSWERS, responses=[httpx.ConnectError, 503, 500])
    with _client(fake, on_attempt=attempts.append) as client:
        with pytest.raises(DecisionUnavailableError):
            client.decide(STATE, {"q1": NOUL}, budget="batch")
    assert [a.attempt for a in attempts] == [1, 2, 3]
    assert [a.status for a in attempts] == [None, 503, 500]
    assert attempts[0].error_type == "transport"
