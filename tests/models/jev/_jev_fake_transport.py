"""A scripted, offline stand-in for the Jev decision service.

``FakeJev`` builds an :class:`httpx.MockTransport` that answers Jev requests
from a script, so every engine, catalog, wiring and paradigm test runs without
touching the network. All data it produces is synthetic and labelled so.

What can be scripted:

* **answers** per question id: a fixed answer dict, or a callable
  ``fn(question, body) -> dict`` that sees the wire question and the whole
  request body. A question id with no script is left out of the reply, which
  is how a service that drops an answer looks on the wire.
* **responses** per attempt, as an ordered sequence consumed one entry per
  request: ``None`` (answer normally), an ``int`` status, a :class:`Reply`, or
  an exception class or instance to raise. Once the sequence is exhausted every
  request is answered normally.
* **rules** added with :meth:`FakeJev.reject_if`: a predicate over the request
  body that, when true, turns the reply into the given status and body. A
  scripted non-``None`` response for the attempt wins over the rules.
* the served ``model`` id (default: echo the requested model) and ``usage``
  (a dict, or a callable over the body).
* **latency** in seconds (a number, or a callable over the body). The fake
  clock advances by ``min(latency, read timeout)``; when latency exceeds the
  request's read timeout the handler raises :class:`httpx.ReadTimeout`, since
  ``MockTransport`` enforces no timeout of its own.

``fake.clock`` and ``fake.sleep`` are the pair a client under test takes:
``sleep`` advances the clock and never sleeps for real. ``dwell`` is the one
real delay, spent inside the handler so concurrent requests overlap and the
peak in-flight count is observable.

Every request body is recorded (``fake.requests``), with the call count and the
peak number of requests in flight, all under one ``threading.Lock``.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

__all__ = [
    "SYNTHETIC_REQUEST_ID_PREFIX",
    "FakeClock",
    "FakeJev",
    "Reply",
    "body_tokens",
    "tokens_over",
]

# Request ids the fake stamps on every reply; synthetic, never a real service id.
SYNTHETIC_REQUEST_ID_PREFIX = "synthetic-req-"

# Header the service uses for its request id (mirrors osprey.models.jev.errors).
_REQUEST_ID_HEADER = "x-typesafe-request-id"

# Output tokens charged per returned answer in the default synthetic usage.
_OUTPUT_TOKENS_PER_ANSWER = 5

Body = dict[str, Any]
AnswerScript = Mapping[str, Any] | Callable[[Mapping[str, Any], Body], Mapping[str, Any]]
Predicate = Callable[[Body], bool]


def body_tokens(body: Mapping[str, Any]) -> int:
    """A crude synthetic token count for a request body: serialised chars / 4, rounded up."""
    return math.ceil(len(json.dumps(body, separators=(",", ":"))) / 4)


def tokens_over(limit: int) -> Predicate:
    """Predicate true when :func:`body_tokens` of the body exceeds ``limit``."""
    return lambda body: body_tokens(body) > limit


@dataclass(frozen=True)
class Reply:
    """One scripted reply. ``body=None`` with status 200 means the normal answer body."""

    status: int = 200
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    latency: float | None = None


class FakeClock:
    """A monotonic clock that only moves when told to; thread-safe."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock never runs backwards")
        with self._lock:
            self._now += seconds


@dataclass(frozen=True)
class _Rule:
    predicate: Predicate
    status: int
    body: Any


class FakeJev:
    """Scripted Jev service behind an ``httpx.MockTransport``."""

    def __init__(
        self,
        answers: Mapping[str, AnswerScript] | None = None,
        *,
        responses: Iterable[Any] = (),
        model: str | None = None,
        usage: Mapping[str, Any] | Callable[[Body], Mapping[str, Any]] | None = None,
        latency: float | Callable[[Body], float] = 0.0,
        dwell: float = 0.0,
        clock_start: float = 1000.0,
    ) -> None:
        self.answers: dict[str, AnswerScript] = dict(answers or {})
        self.responses: list[Any] = list(responses)
        self.model = model
        self.usage = usage
        self.latency = latency
        self.dwell = dwell
        self.clock = FakeClock(clock_start)
        self.requests: list[Body] = []
        self.call_count = 0
        self.peak_in_flight = 0
        self._in_flight = 0
        self._rules: list[_Rule] = []
        self._lock = threading.Lock()
        self.transport = httpx.MockTransport(self._handle)

    # -- scripting -----------------------------------------------------------

    def reject_if(self, predicate: Predicate, *, status: int = 400, body: Any = None) -> FakeJev:
        """Answer ``status``/``body`` whenever ``predicate(request_body)`` is true."""
        self._rules.append(_Rule(predicate, status, body))
        return self

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock; never sleeps for real."""
        self.clock.advance(seconds)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    # -- handler -------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body: Body = json.loads(request.content or b"{}")
        with self._lock:
            attempt = self.call_count
            self.call_count += 1
            self.requests.append(body)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self.dwell > 0:
                time.sleep(self.dwell)
            return self._reply(request, body, attempt)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _reply(self, request: httpx.Request, body: Body, attempt: int) -> httpx.Response:
        scripted = self.responses[attempt] if attempt < len(self.responses) else None

        if isinstance(scripted, type) and issubclass(scripted, BaseException):
            self._pass_time(request, body, None)
            if issubclass(scripted, httpx.RequestError):
                raise scripted("synthetic scripted failure", request=request)
            raise scripted("synthetic scripted failure")
        if isinstance(scripted, BaseException):
            self._pass_time(request, body, None)
            raise scripted

        if isinstance(scripted, int):
            scripted = Reply(status=scripted)
        if scripted is None:
            for rule in self._rules:
                if rule.predicate(body):
                    scripted = Reply(status=rule.status, body=rule.body)
                    break
        if not isinstance(scripted, Reply | None):
            raise TypeError(f"unscriptable response entry: {scripted!r}")
        reply = scripted if scripted is not None else Reply()

        self._pass_time(request, body, reply.latency)

        headers = {_REQUEST_ID_HEADER: f"{SYNTHETIC_REQUEST_ID_PREFIX}{attempt + 1}"}
        headers.update(reply.headers)
        if reply.status == 200 and reply.body is None:
            return httpx.Response(200, json=self._answer_body(body), headers=headers)
        if reply.body is None:
            return httpx.Response(reply.status, headers=headers)
        if isinstance(reply.body, bytes | str):
            return httpx.Response(reply.status, content=reply.body, headers=headers)
        return httpx.Response(reply.status, json=reply.body, headers=headers)

    def _pass_time(self, request: httpx.Request, body: Body, override: float | None) -> None:
        if override is not None:
            latency = override
        elif callable(self.latency):
            latency = self.latency(body)
        else:
            latency = self.latency
        timeout = (request.extensions.get("timeout") or {}).get("read")
        if timeout is not None and latency > timeout:
            self.clock.advance(timeout)
            raise httpx.ReadTimeout("synthetic read timeout", request=request)
        self.clock.advance(latency)

    def _answer_body(self, body: Body) -> Body:
        questions: Mapping[str, Any] = body.get("questions") or {}
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            script = self.answers.get(qid)
            if script is None:
                continue
            answers[qid] = dict(script(question, body) if callable(script) else script)
        if callable(self.usage):
            usage = dict(self.usage(body))
        elif self.usage is not None:
            usage = dict(self.usage)
        else:
            usage = {
                "input_tokens": body_tokens(body),
                "output_tokens": _OUTPUT_TOKENS_PER_ANSWER * len(answers),
            }
        return {
            "model": self.model if self.model is not None else body.get("model", ""),
            "answers": answers,
            "usage": usage,
        }
