"""The synchronous Jev decision client.

``JevClient`` holds one ``httpx.Client`` and is safe to share across threads:
FastMCP runs sync tools in a thread pool, so there is no async client.

Construction never raises for a missing key. The route (base URL, key, model)
is resolved on the first ``decide`` call, before any HTTP, so a missing key
comes out of ``decide`` as ``DecisionAuthError(error_type="no_key")`` and the
caller maps it per call. A resolved route is kept; a failed resolution is
retried on the next call, so a key set later is picked up.

One ``decide`` call:

1. validates the request locally (non-empty instructions, criteria shapes,
   option and level counts, the configured ``max_choice_options``) and raises
   ``DecisionRequestError(error_type="invalid_request")`` before any HTTP;
2. POSTs the body, with ``model`` always sent, to the route's endpoint;
3. maps a non-200 reply, a 200 carrying an ``error`` object and an undecodable
   or invalid 200 through ``error_from_response``;
4. validates and normalises the answers and checks every requested id came
   back, raising ``IncompleteAnswersError`` (carrying the billed call) if not;
5. returns a ``JevResult`` with one ``JevCall`` per request sent.

Batching: there is no cap on the number of questions. A request whose state
plus its longest question is estimated above the route's
``max_state_plus_question_tokens`` is refused locally with
``DecisionRequestError(error_type="state_too_large")``. Otherwise the questions
are packed, in order, into same-state chunks that each fit
``max_request_tokens``; every chunk carries at least one question. Chunks run
on a per-call thread pool of ``max_concurrency`` workers, and one client-wide
semaphore of ``max_concurrency`` permits is held around each HTTP send only
(never across a backoff wait), so the number of requests in flight from one
client never exceeds ``max_concurrency`` however many threads call ``decide``.
Every chunk and every retry shares one deadline: the caller's, else one minted
per call; a chunk not started by then fails with
``DecisionUnavailableError(error_type="deadline_exceeded")`` without HTTP. A
multi-question chunk answered with 413 or ``max_tokens_exceeded`` is split in
half once; a one-question request raises. Each chunk is checked for missing
answers on its own, the answers are merged and ``JevResult.calls`` keeps every
chunk's call. A failure raised after other chunks were billed carries those
chunks' calls in ``completed_calls``.

Cost: ``billed_cost_usd`` is the OpenRouter ``usage.cost`` and sets
``cost_source="billed"``; otherwise the cost is ``computed``. The computed and
list costs always come from ``api.jev.input_cost_per_million_tokens``.

Every ``DecisionError`` raised from a parse, validation or transport failure is
raised ``from None``: a pydantic ``ValidationError`` repeats its ``input_value``
and an error body can echo the state, so neither may ride on ``__cause__`` or
``__context__`` into a traceback. Bodies are never logged or stored.

Budgets (``api.jev.budgets``): every attempt's timeout is ``timeout_s``
clipped to what is left of the deadline, and ``JevCall.attempts`` counts every
attempt made. A failure is retried, up to ``max_retries`` times, only when it
is a timeout, a connection failure or a 5xx; the ``batch`` budget also retries
a 429. The wait before a retry is ``retry-after-ms`` or ``retry-after``
(seconds) when the reply carries one, else exponential backoff; the
``interactive`` budget caps every wait at one second. A retry whose wait would
reach the deadline is not made and the last error is raised. Waits and
deadlines run on the injected ``clock`` and ``sleep`` only.

``on_attempt`` receives one ``JevAttempt`` per HTTP attempt, success or
failure, never with a body; calls are serialised under the client's own lock.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from osprey.models.jev.config import Budget, JevSettings, ResolvedRoute, resolve_route
from osprey.models.jev.errors import (
    REQUEST_ID_HEADER,
    DecisionError,
    DecisionRequestError,
    DecisionUnavailableError,
    IncompleteAnswersError,
    error_from_response,
)
from osprey.models.jev.models import (
    ChoiceQuestion,
    JevAttempt,
    JevCall,
    JevRequest,
    JevResponse,
    JevResult,
)
from osprey.models.jev.tokens import (
    estimate_question,
    estimate_request,
    estimate_state_plus_longest,
)

__all__ = ["BUDGET_NAMES", "JevClient"]

#: The budget names ``decide`` and ``new_deadline`` accept.
BUDGET_NAMES: tuple[str, ...] = ("interactive", "batch")

_TOKENS_PER_MILLION = 1_000_000

#: The first backoff wait, doubled on every further retry.
_BACKOFF_BASE_S = 0.5
#: The longest wait the ``interactive`` budget spends before a retry.
_INTERACTIVE_MAX_WAIT_S = 1.0
_RETRY_AFTER_HEADER = "retry-after"
_RETRY_AFTER_MS_HEADER = "retry-after-ms"
_TOO_MANY_REQUESTS = 429
_PAYLOAD_TOO_LARGE = 413
_MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"


@dataclass(frozen=True)
class _Failure:
    """One failed attempt: the error to raise and the server's requested wait, if any."""

    error: DecisionError
    status: int | None
    retry_after_s: float | None


@dataclass(frozen=True)
class _ChunkResult:
    """The merged answers of one chunk and every call it was billed for."""

    answers: dict[str, Any]
    served_model: str | None
    calls: list[JevCall]


def _too_large(error: DecisionError) -> bool:
    """Whether the service refused a request for its size."""
    return error.status == _PAYLOAD_TOO_LARGE or error.error_type == _MAX_TOKENS_EXCEEDED


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """The wait a reply asks for, in seconds; ``None`` when absent or unusable.

    ``retry-after-ms`` wins over ``retry-after``; an HTTP-date is not used,
    since it reads the wall clock rather than the injected one.
    """
    for name, scale in ((_RETRY_AFTER_MS_HEADER, 1000.0), (_RETRY_AFTER_HEADER, 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip()) / scale
        except ValueError:
            continue
        if value >= 0 and value == value and value != float("inf"):
            return value
    return None


def _validation_summary(exc: ValidationError) -> str:
    """``loc: msg`` of each validation error; ``input`` is never read."""
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        loc = ".".join(str(p) for p in error.get("loc", ()))
        parts.append(
            f"{loc}: {error.get('msg', 'invalid')}" if loc else error.get("msg", "invalid")
        )
    return "; ".join(parts) or "invalid request"


def _invalid(message: str) -> DecisionRequestError:
    return DecisionRequestError(message, error_type="invalid_request")


class JevClient:
    """Synchronous client for the Jev structured-decision endpoint."""

    def __init__(
        self,
        settings: JevSettings,
        config: Mapping[str, Any],
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_attempt: Callable[[JevAttempt], None] | None = None,
    ) -> None:
        """Bind the settings and config; ``transport`` and ``clock`` are injectable for tests."""
        self._settings = settings
        self._config = config
        self._clock = clock
        self._sleep = sleep
        self._on_attempt = on_attempt
        self._http = httpx.Client(transport=transport, follow_redirects=False)
        self._route: ResolvedRoute | None = None
        self._route_lock = threading.Lock()
        self._attempt_lock = threading.Lock()
        # Bounds the requests in flight across every thread using this client.
        self._inflight = threading.BoundedSemaphore(settings.max_concurrency)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> JevClient:
        """Enter a context that closes the client on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the underlying HTTP client."""
        self.close()

    # -- public API ----------------------------------------------------------

    @property
    def settings(self) -> JevSettings:
        """The validated ``api.jev`` settings this client uses."""
        return self._settings

    def new_deadline(self, budget: str) -> float:
        """A deadline ``budget.total_s`` seconds from now on the injected clock."""
        return self._clock() + self._budget(budget).total_s

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        budget: str,
        deadline: float | None = None,
    ) -> JevResult:
        """Ask every question about ``state`` and return the normalised answers.

        Only ``DecisionError`` subclasses escape.
        """
        route = self._resolve()
        limits = self._budget(budget)
        request = self._build_request(route, state, questions)
        if deadline is None:
            deadline = self._clock() + limits.total_s
        return self._decide_batched(request, route, budget, limits, deadline)

    # -- request building ----------------------------------------------------

    def _resolve(self) -> ResolvedRoute:
        with self._route_lock:
            if self._route is None:
                self._route = resolve_route(self._settings, self._config)
            return self._route

    def _budget(self, name: str) -> Budget:
        if name not in BUDGET_NAMES:
            raise _invalid(f"unknown budget {name!r}; expected one of {', '.join(BUDGET_NAMES)}")
        limits: Budget = getattr(self._settings.budgets, name)
        return limits

    def _build_request(
        self, route: ResolvedRoute, state: Any, questions: Mapping[str, Any]
    ) -> JevRequest:
        """Validate the request locally; raise ``invalid_request`` before any HTTP."""
        if not isinstance(questions, Mapping):
            raise _invalid("questions must be a mapping of question id to question")
        payload = {
            "model": route.model,
            "state": state,
            "questions": {
                qid: q.model_dump(by_alias=True) if hasattr(q, "model_dump") else q
                for qid, q in questions.items()
            },
        }
        # Raised outside the handler so the ValidationError is not even __context__.
        summary: str | None = None
        try:
            request = JevRequest.model_validate(payload)
        except ValidationError as exc:
            summary = _validation_summary(exc)
        if summary is not None:
            raise _invalid(f"invalid Jev request: {summary}") from None
        cap = self._settings.max_choice_options
        for qid, question in request.questions.items():
            if isinstance(question, ChoiceQuestion) and len(question.criteria) > cap:
                raise _invalid(
                    f"invalid Jev request: question {qid!r} has {len(question.criteria)} options,"
                    f" above max_choice_options ({cap})"
                )
        return request

    @staticmethod
    def _wire_body(request: JevRequest) -> dict[str, Any]:
        body = request.model_dump(mode="json", by_alias=True)
        for question in body["questions"].values():
            if question.get("criteria") is None:
                question.pop("criteria", None)
        return body

    # -- batching ------------------------------------------------------------

    def _decide_batched(
        self,
        request: JevRequest,
        route: ResolvedRoute,
        budget: str,
        limits: Budget,
        deadline: float,
    ) -> JevResult:
        """Refuse, split, send every chunk under one deadline, and merge the answers."""
        route_limits = self._settings.active_route
        if (
            estimate_state_plus_longest(request.state, request.questions)
            > route_limits.max_state_plus_question_tokens
        ):
            raise DecisionRequestError(
                "the state plus its longest question is estimated above"
                f" max_state_plus_question_tokens ({route_limits.max_state_plus_question_tokens})",
                error_type="state_too_large",
            )
        chunks = self._chunks(request, route_limits.max_request_tokens)

        def run(index: int) -> _ChunkResult:
            return self._run_chunk(chunks[index], route, budget, limits, deadline, index)

        outcomes: list[_ChunkResult | DecisionError | None] = [None] * len(chunks)
        if len(chunks) == 1:
            try:
                outcomes[0] = run(0)
            except DecisionError as exc:
                outcomes[0] = exc
        else:
            workers = min(self._settings.max_concurrency, len(chunks))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run, index): index for index in range(len(chunks))}
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    failed = future.exception()
                    if failed is None:
                        outcomes[futures[future]] = future.result()
                        continue
                    # Chunks not yet started are dropped; running ones finish.
                    for other in futures:
                        other.cancel()
                    if not isinstance(failed, DecisionError):
                        raise failed
                    outcomes[futures[future]] = failed

        return self._merge(outcomes, route)

    @staticmethod
    def _merge(
        outcomes: list[_ChunkResult | DecisionError | None], route: ResolvedRoute
    ) -> JevResult:
        """One result from every chunk, or the first failure carrying every billed call."""
        calls: list[JevCall] = []
        failure: DecisionError | None = None
        for outcome in outcomes:
            if isinstance(outcome, DecisionError):
                calls.extend(outcome.completed_calls)
                if failure is None:
                    failure = outcome
            elif outcome is not None:
                calls.extend(outcome.calls)
        if failure is not None:
            failure.completed_calls = calls
            raise failure from None
        answers: dict[str, Any] = {}
        served_model: str | None = None
        for outcome in outcomes:
            assert isinstance(outcome, _ChunkResult)
            answers.update(outcome.answers)
            if served_model is None:
                served_model = outcome.served_model
        return JevResult(
            answers=answers,
            requested_model=route.model,
            served_model=served_model,
            route=route.route,
            calls=calls,
        )

    @staticmethod
    def _chunks(request: JevRequest, max_request_tokens: int) -> list[JevRequest]:
        """Pack the questions, in order, into same-state requests within ``max_request_tokens``.

        Each chunk holds at least one question, so a lone question that passed
        the state-plus-question refusal is sent alone.
        """
        base = estimate_request(request.state, {})
        groups: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        used = base
        for qid, question in request.questions.items():
            cost = estimate_question(qid, question)
            if current and used + cost > max_request_tokens:
                groups.append(current)
                current, used = {}, base
            current[qid] = question
            used += cost
        groups.append(current)
        if len(groups) == 1:
            return [request]
        return [request.model_copy(update={"questions": group}) for group in groups]

    def _run_chunk(
        self,
        chunk: JevRequest,
        route: ResolvedRoute,
        budget: str,
        limits: Budget,
        deadline: float,
        chunk_index: int,
    ) -> _ChunkResult:
        """Send one chunk; bisect it once when the service finds it too large."""
        try:
            response, call = self._send_with_budget(
                chunk, route, budget, limits, deadline, chunk_index=chunk_index
            )
        except DecisionError as exc:
            if len(chunk.questions) < 2 or not _too_large(exc):
                raise
            billed = list(exc.completed_calls)
        else:
            return _ChunkResult(dict(response.answers), response.model, [call])

        ids = list(chunk.questions)
        middle = len(ids) // 2
        answers: dict[str, Any] = {}
        served_model: str | None = None
        for half_ids in (ids[:middle], ids[middle:]):
            half = chunk.model_copy(update={"questions": {q: chunk.questions[q] for q in half_ids}})
            try:
                response, call = self._send_with_budget(
                    half, route, budget, limits, deadline, chunk_index=chunk_index
                )
            except DecisionError as exc:
                exc.completed_calls = billed + exc.completed_calls
                raise
            billed.append(call)
            answers.update(response.answers)
            if served_model is None:
                served_model = response.model
        return _ChunkResult(answers, served_model, billed)

    # -- sending -------------------------------------------------------------

    def _send_with_budget(
        self,
        request: JevRequest,
        route: ResolvedRoute,
        budget: str,
        limits: Budget,
        deadline: float,
        *,
        chunk_index: int,
    ) -> tuple[JevResponse, JevCall]:
        """Attempt, classify, wait and retry within ``limits`` and ``deadline``."""
        interactive = budget == "interactive"
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise DecisionUnavailableError(
                    "the Jev deadline passed before the request started",
                    error_type="deadline_exceeded",
                )
            timeout = min(limits.timeout_s, remaining)
            outcome = self._attempt(
                request, route, timeout, chunk_index=chunk_index, attempt=attempt
            )
            if not isinstance(outcome, _Failure):
                return outcome
            if attempt > limits.max_retries or not self._retryable(outcome, interactive):
                raise outcome.error from None
            wait = self._wait(outcome, attempt, interactive)
            if self._clock() + wait >= deadline:
                raise outcome.error from None
            self._sleep(wait)

    @staticmethod
    def _retryable(failure: _Failure, interactive: bool) -> bool:
        """Timeouts, connection failures and 5xx; a 429 only outside ``interactive``."""
        if failure.error.error_type in ("timeout", "transport"):
            return True
        status = failure.status
        if status is None:
            return False
        if status == _TOO_MANY_REQUESTS:
            return not interactive
        return status >= 500

    @staticmethod
    def _wait(failure: _Failure, attempt: int, interactive: bool) -> float:
        """The server's requested wait, else exponential backoff; capped when interactive."""
        if failure.retry_after_s is not None:
            wait = failure.retry_after_s
        else:
            wait = _BACKOFF_BASE_S * 2 ** (attempt - 1)
        return min(wait, _INTERACTIVE_MAX_WAIT_S) if interactive else wait

    def _attempt(
        self,
        request: JevRequest,
        route: ResolvedRoute,
        timeout: float,
        *,
        chunk_index: int,
        attempt: int,
    ) -> tuple[JevResponse, JevCall] | _Failure:
        """One HTTP attempt: send, map, parse; report it through ``on_attempt``.

        A failure is returned, not raised, with the reply's requested wait.
        """
        body = self._wire_body(request)
        started = self._clock()
        try:
            with self._inflight:
                started = self._clock()
                http_response = self._http.post(
                    route.endpoint,
                    json=body,
                    headers={"Authorization": f"Bearer {route.api_key}"},
                    timeout=httpx.Timeout(timeout),
                )
        except httpx.TimeoutException:
            error: DecisionError = DecisionUnavailableError(
                f"Jev request timed out after {timeout:g} s", error_type="timeout"
            )
        except httpx.RequestError as exc:
            error = DecisionUnavailableError(
                f"Jev request failed in transport: {type(exc).__name__}", error_type="transport"
            )
        else:
            latency = self._clock() - started
            status = http_response.status_code
            failure: _Failure | None = None
            try:
                response, call = self._parse(
                    http_response, route, request, latency=latency, attempts=attempt
                )
            except DecisionError as exc:
                failure = _Failure(exc, status, _parse_retry_after(http_response.headers))
            if failure is not None:
                self._report(chunk_index, attempt, status, failure.error, latency)
                return failure
            self._report(chunk_index, attempt, 200, None, latency, call=call)
            return response, call

        self._report(chunk_index, attempt, None, error, self._clock() - started)
        return _Failure(error, None, None)

    def _parse(
        self,
        http_response: httpx.Response,
        route: ResolvedRoute,
        request: JevRequest,
        *,
        latency: float,
        attempts: int,
    ) -> tuple[JevResponse, JevCall]:
        status = http_response.status_code
        headers = http_response.headers
        content = http_response.content
        if status != 200:
            raise error_from_response(status, headers, content)
        try:
            payload = json.loads(content)
        except ValueError:
            payload = None
        if not isinstance(payload, Mapping) or "error" in payload:
            raise error_from_response(status, headers, content)
        # The raise stays outside the except block: the typed error must carry no
        # __context__, or the ValidationError with the raw body would chain onto it.
        response: JevResponse | None = None
        try:  # noqa: SIM105
            response = JevResponse.model_validate(payload)
        except ValidationError:
            pass
        if response is None:
            raise error_from_response(status, headers, content) from None

        request_id = headers.get(REQUEST_ID_HEADER) or response.id
        call = self._call(
            response, route, latency=latency, attempts=attempts, request_id=request_id
        )
        missing = sorted(set(request.questions) - set(response.answers))
        if missing:
            raise IncompleteAnswersError(
                f"reply omitted answers for: {', '.join(missing)}",
                status=status,
                request_id=request_id,
                completed_calls=[call],
            )
        return response, call

    def _call(
        self,
        response: JevResponse,
        route: ResolvedRoute,
        *,
        latency: float,
        attempts: int,
        request_id: str | None,
    ) -> JevCall:
        usage = response.usage
        list_cost = (
            usage.input_tokens * self._settings.input_cost_per_million_tokens / _TOKENS_PER_MILLION
        )
        billed = usage.cost if route.route == "openrouter" else None
        return JevCall(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            billed_cost_usd=billed,
            cost_usd=billed if billed is not None else list_cost,
            cost_source="billed" if billed is not None else "computed",
            list_cost_usd=list_cost,
            latency_s=latency,
            served_model=response.model,
            request_id=request_id,
            attempts=attempts,
        )

    def _report(
        self,
        chunk_index: int,
        attempt: int,
        status: int | None,
        error: DecisionError | None,
        latency: float,
        *,
        call: JevCall | None = None,
    ) -> None:
        if self._on_attempt is None:
            return
        if call is None and error is not None and error.completed_calls:
            call = error.completed_calls[-1]
        record = JevAttempt(
            chunk_index=chunk_index,
            attempt=attempt,
            status=status,
            error_type=error.error_type if error is not None else None,
            latency_s=latency,
            input_tokens=call.input_tokens if call else None,
            output_tokens=call.output_tokens if call else None,
            billed_cost_usd=call.billed_cost_usd if call else None,
            served_model=call.served_model if call else None,
            request_id=call.request_id if call else (error.request_id if error else None),
        )
        with self._attempt_lock:
            self._on_attempt(record)
