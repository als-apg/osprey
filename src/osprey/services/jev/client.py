"""Async client for TypeSafe's System One API — the ``jev`` model.

System One is not a chat endpoint. A request carries one **state** (the JSON the
decision is about) and a map of **typed questions** asked against it; the reply
carries one typed answer per question. There is no prose to parse and no schema
to coax out of a completion, which is what makes it usable on a latency budget
an operator can feel: the questions are evaluated in parallel and in isolation,
so asking thirty of them costs about what asking one costs.

Three properties of this client are deliberate rather than incidental:

**Unconfigured is a state, not a failure.** A deployment that never set an API
key is a supported configuration — the ``jev`` search module is opt-in and every
caller has a fallback path. Such a client reports itself unavailable without
ever opening a socket, and says nothing about it in the log.

**The caller owns staleness, this client owns concurrency.** Per-keystroke
callers supersede their own requests; that is a decision only the caller can
make, so nothing here cancels anything. What the client does own is the number
of requests allowed to be in flight at once, because that is a property of the
endpoint rather than of any one caller.

**A retry is for the two statuses that mean "later".** ``429`` (rate limited)
and ``529`` (overloaded) are retried with exponential backoff; every other 4xx
is the request's own fault and is raised immediately, because retrying a
malformed question just spends the budget twice.

Typical use, with the caller gating on availability and taking its own fallback
path when no key is configured::

    client = JevClient(JevSettings())
    if client.is_available():
        result = await client.ask(
            state={"query": "beam loss", "candidates": [...]},
            questions={"c0": score_question("How relevant?", RELEVANCE_LEVELS)},
        )
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger("osprey.services.jev")

#: The System One endpoint. Overridable so a facility can point the module at a
#: gateway of its own without patching code.
DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

#: The model alias. ``jev-latest`` tracks the newest revision; a deployment that
#: wants a frozen one pins the exact name in its settings block.
DEFAULT_MODEL = "jev-latest"

#: Environment variable the API key is read from.
DEFAULT_API_KEY_ENV = "TYPESAFE_API_KEY"

#: Per-request ceiling. Generous relative to the ~150 ms this endpoint answers
#: in: it is a guard against a hung socket, not a latency budget. The caller's
#: own budget is the caller's to enforce.
DEFAULT_TIMEOUT_SECONDS = 5.0

#: How many requests may be in flight at once. A per-keystroke caller can
#: otherwise open one connection per character typed.
DEFAULT_MAX_IN_FLIGHT = 4

#: Attempts after the first, for the two statuses that mean "later".
DEFAULT_MAX_RETRIES = 2

#: First backoff step, doubled per retry and jittered.
_BACKOFF_BASE_SECONDS = 0.15

#: Statuses worth retrying: rate limited, and the service's own overload code.
_RETRY_STATUSES = frozenset({429, 529})

#: The three question types System One accepts.
QuestionType = Literal["noul", "choice", "score"]


class JevError(RuntimeError):
    """The endpoint was reached but could not answer the request."""


class JevUnavailableError(JevError):
    """No API key is configured, or the endpoint could not be reached at all.

    Separate from :class:`JevError` because it is the one failure a caller is
    expected to have a plain fallback for: it means "this deployment does not
    have Jev", not "this request was wrong".
    """


@dataclass(frozen=True)
class JevSettings:
    """Endpoint and transport knobs.

    Attributes:
        endpoint: Full URL of the System One endpoint.
        model: Model alias sent with every request.
        api_key_env: Environment variable the API key is read from.
        timeout_seconds: Per-request timeout.
        max_in_flight: Requests allowed to be in flight at once.
        max_retries: Retries after the first attempt, for ``429``/``529`` only.
    """

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    max_retries: int = DEFAULT_MAX_RETRIES


@dataclass(frozen=True)
class JevAnswer:
    """One typed answer.

    The three answer shapes are normalized onto one object so a caller reading
    a score does not need a different accessor from a caller reading a choice.

    Attributes:
        type: Which question type produced this answer.
        value: The answer itself — the level for ``score``, the chosen option
            for ``choice``, the 0-1 truth value for ``noul``.
        confidence: The model's confidence in ``value``, 0-1. ``noul`` answers
            carry none and report ``None``; their value *is* a probability.
        probabilities: Per-option (or per-level) probabilities when the answer
            type carries them, else empty.
        legend: For ``score``, the level-to-description mapping the model
            scored against. Empty for the other types.
    """

    type: QuestionType
    value: Any
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    legend: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JevAnswer:
        """Build an answer from one entry of a response's ``answers`` map.

        Args:
            payload: The raw answer object.

        Returns:
            The normalized answer.

        Raises:
            JevError: If the payload names no known answer type.
        """
        kind = payload.get("type")
        if kind not in ("noul", "choice", "score"):
            raise JevError(f"Unknown answer type {kind!r} in System One response")
        value = payload.get(kind)
        confidence = payload.get("confidence")
        return cls(
            type=kind,
            value=value,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            probabilities=dict(payload.get("probabilities") or {}),
            legend=dict(payload.get("legend") or {}),
        )


@dataclass(frozen=True)
class JevResult:
    """A decoded System One reply.

    Attributes:
        answers: One answer per question, keyed by the question id the caller
            sent.
        model: The model that answered, as reported.
        input_tokens: Reported input tokens, or 0 when the reply omitted usage.
        output_tokens: Reported output tokens, or 0 when the reply omitted it.
        latency_ms: Wall-clock time the call took, including any retry waits.
    """

    answers: Mapping[str, JevAnswer]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


def score_question(instructions: str, criteria: Sequence[str]) -> dict[str, Any]:
    """Build a ``score`` question — rate the state against ordered levels.

    Args:
        instructions: What is being rated, in one sentence.
        criteria: Level descriptions, weakest first. At least two.

    Returns:
        The question object, ready to place in a request's ``questions`` map.

    Raises:
        ValueError: If fewer than two levels were supplied.
    """
    levels = list(criteria)
    if len(levels) < 2:
        raise ValueError("A score question needs at least two criteria levels")
    return {"type": "score", "instructions": instructions, "criteria": levels}


def choice_question(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
    """Build a ``choice`` question — pick one option from a named set.

    Args:
        instructions: The decision to make, in one sentence.
        criteria: Option name to a description of when it applies.

    Returns:
        The question object.

    Raises:
        ValueError: If fewer than two options were supplied.
    """
    options = dict(criteria)
    if len(options) < 2:
        raise ValueError("A choice question needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul_question(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build a ``noul`` question — how true is this, as a 0-1 value.

    Args:
        instructions: The proposition to evaluate.
        criteria: Optional ``{"true": ..., "false": ...}`` descriptions.

    Returns:
        The question object.
    """
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = dict(criteria)
    return question


class JevClient:
    """Asks typed questions of one System One endpoint.

    The client is safe to share across concurrent callers: the HTTP connection
    pool is reused and the in-flight cap is shared, which is the point of
    sharing one.
    """

    def __init__(self, settings: JevSettings | None = None, api_key: str | None = None) -> None:
        """Build a client.

        Args:
            settings: Endpoint and transport knobs. Omitted, the defaults.
            api_key: The key to send. Omitted, it is read from the environment
                variable named by ``settings.api_key_env`` at construction, so
                a key exported after this point is not picked up — deliberate,
                so availability is a fixed property of a built client.
        """
        self.settings = settings or JevSettings()
        self._api_key = api_key or os.environ.get(self.settings.api_key_env) or None
        self._semaphore = asyncio.Semaphore(max(1, self.settings.max_in_flight))
        self._client: Any | None = None

    def is_available(self) -> bool:
        """Whether this client has a key to authenticate with.

        Returns:
            True when a key was supplied or found in the environment. It says
            nothing about whether the endpoint is reachable — that is only
            knowable by asking it.
        """
        return bool(self._api_key)

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> JevResult:
        """Ask every question against one state, in a single request.

        Args:
            state: The JSON-serializable object the questions are about.
            questions: Question id to question object. Ids are echoed back as
                the keys of the result's ``answers``.

        Returns:
            The decoded reply.

        Raises:
            JevUnavailableError: If no API key is configured, or the endpoint
                could not be reached.
            JevError: If the endpoint answered with an error status, or with a
                body this client cannot decode.
        """
        if not self._api_key:
            raise JevUnavailableError(
                f"No Jev API key: set {self.settings.api_key_env} in the environment"
            )
        if not questions:
            raise ValueError("ask() needs at least one question")

        payload = {"model": self.settings.model, "state": state, "questions": dict(questions)}
        started = time.monotonic()
        async with self._semaphore:
            body = await self._post_with_retries(payload)
        latency_ms = int((time.monotonic() - started) * 1000)

        raw_answers = body.get("answers")
        if not isinstance(raw_answers, dict):
            raise JevError("System One response carried no 'answers' map")
        usage = body.get("usage") or {}
        return JevResult(
            answers={key: JevAnswer.from_payload(value) for key, value in raw_answers.items()},
            model=str(body.get("model") or self.settings.model),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool, if one was opened."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # --- Internal ---

    def _http(self) -> Any:
        """Return the shared ``httpx.AsyncClient``, building it on first use.

        Built lazily so that constructing a client — which the search module
        does on every deployment, enabled or not — opens no pool and needs no
        running event loop.

        Returns:
            The client.
        """
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.settings.timeout_seconds)
        return self._client

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST the payload, retrying only the statuses that mean "later".

        Args:
            payload: The request body.

        Returns:
            The decoded response body.

        Raises:
            JevUnavailableError: If the endpoint could not be reached on the
                final attempt.
            JevError: If it answered with an error status, or an undecodable
                body.
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(0, self.settings.max_retries) + 1
        last_status: int | None = None
        last_detail = ""

        for attempt in range(attempts):
            try:
                response = await self._http().post(
                    self.settings.endpoint, json=payload, headers=headers
                )
            except httpx.HTTPError as exc:
                if attempt + 1 >= attempts:
                    raise JevUnavailableError(
                        f"Could not reach {self.settings.endpoint}: {exc}"
                    ) from exc
                await self._backoff(attempt)
                continue

            if response.status_code < 300:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise JevError("System One answered with a body that is not JSON") from exc
                if not isinstance(body, dict):
                    raise JevError("System One answered with a body that is not an object")
                return body

            last_status = response.status_code
            last_detail = response.text[:500]
            if response.status_code not in _RETRY_STATUSES or attempt + 1 >= attempts:
                break
            await self._backoff(attempt)

        raise JevError(f"System One returned HTTP {last_status}: {last_detail}")

    @staticmethod
    async def _backoff(attempt: int) -> None:
        """Sleep the jittered exponential step for a completed attempt.

        Args:
            attempt: Zero-based index of the attempt that just failed.
        """
        delay = _BACKOFF_BASE_SECONDS * (2**attempt)
        await asyncio.sleep(delay + random.uniform(0, delay / 2))


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MAX_IN_FLIGHT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "JevAnswer",
    "JevClient",
    "JevError",
    "JevResult",
    "JevSettings",
    "JevUnavailableError",
    "QuestionType",
    "choice_question",
    "noul_question",
    "score_question",
]
