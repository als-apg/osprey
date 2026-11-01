"""The decision engine: a Jev model picks among the facility's verdict options.

One :class:`~osprey.models.jev.models.ChoiceQuestion` carries the watchdog's
instructions and its options under neutral ids (``k001``, ``k002``, … in
definition order) plus a reserved ``none-fits``, so a facility's own option ids
never reach the model. The state is the instructions followed by the bounded
session view; :class:`~osprey.services.watchdog.verdict.InputDigest` hashes those
exact state bytes, so the recorded input covers both.

The view's budget is what the route's ``max_state_plus_question_tokens`` leaves
after the question, the request overhead and the instructions in the state,
counted by the estimator the client refuses with, so a long session is cut rather
than refused.

The score is the summed probability of the flag options, and the watchdog flags
at or above its threshold, with the facility text of its likeliest flag option as
the reason. The score is uncalibrated, and a verdict says so.

The engine reads nothing of its own: the nodes are handed in, and the Jev endpoint
is the only thing it talks to. Neither the state nor any Jev error message is ever
logged, since both can repeat the session.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import JevSettings
from osprey.models.jev.errors import DecisionError
from osprey.models.jev.models import PROBABILITY_DECIMALS, ChoiceAnswer, ChoiceQuestion, JevCall
from osprey.models.jev.tokens import (
    REQUEST_OVERHEAD_TOKENS,
    estimate_request,
    estimate_state_plus_longest,
)
from osprey.services.watchdog.config import WatchdogDefinition
from osprey.services.watchdog.history import (
    SessionNodes,
    as_session_nodes,
    build_session_view_from_nodes,
)
from osprey.services.watchdog.verdict import UNBOUND_ID, ErrorType, InputDigest, Verdict

logger = logging.getLogger(__name__)

__all__ = ["BUDGET", "DecisionEngine", "NONE_FITS", "QUESTION_ID", "UNBOUND_ID"]

#: The ``api.jev.budgets`` entry every decision watchdog call runs under.
BUDGET = "interactive"

#: The one question's id on the wire.
QUESTION_ID = "watchdog"

#: The reserved option that lets the model say none of the facility's options fits.
NONE_FITS = "none-fits"

_NONE_FITS_TEXT = "None of the other options describes this session."

#: The Jev error types that mean the time ran out.
_TIMEOUT_ERRORS = frozenset({"timeout", "deadline_exceeded"})


def _state_tokens(text: str) -> int:
    """*text*'s size as state, by the estimator the client's size refusal uses."""
    return estimate_request(text, {}) - REQUEST_OVERHEAD_TOKENS


def _option_key(index: int) -> str:
    return f"k{index + 1:03d}"


def _build_question(defn: WatchdogDefinition, instructions: str) -> ChoiceQuestion:
    criteria: dict[str, Any] = {
        _option_key(index): option.text for index, option in enumerate(defn.options)
    }
    criteria[NONE_FITS] = _NONE_FITS_TEXT
    return ChoiceQuestion(instructions=instructions, criteria=criteria)


def _state_prefix(instructions: str) -> str:
    return f"WATCHDOG INSTRUCTIONS:\n{instructions}\n\nSESSION:\n"


def _total_cost(calls: Sequence[JevCall]) -> float | None:
    return sum(call.cost_usd for call in calls) if calls else None


def _error_type(error: DecisionError) -> ErrorType:
    if error.error_type == "no_key":
        return "missing_key"
    return "timeout" if error.error_type in _TIMEOUT_ERRORS else "decision_error"


class DecisionEngine:
    """Judges prompts through the Jev decision client built from *config*."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Keep the config; the client is built on the first judgement.

        ``transport``, ``clock`` and ``sleep`` are the client's own test seams.
        *clock* must be the clock the ``deadline`` given to :meth:`judge` was read from.
        """
        self._config = config
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._client: JevClient | None = None
        self._client_lock = threading.Lock()

    def close(self) -> None:
        """Close the client, if one was built."""
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    async def judge(
        self,
        defn: WatchdogDefinition,
        nodes: SessionNodes | Sequence[Mapping[str, Any]],
        prompt: str,
        *,
        deadline: float,
        session_key: str | None = None,
        record_session_id: str = UNBOUND_ID,
        submission_id: str = UNBOUND_ID,
    ) -> Verdict:
        """Judge *prompt* in the session *nodes* describe; never raises on a failure.

        *deadline* is a value of the engine's clock, shared by every Jev attempt.
        The three ids name the prompt in the verdict.
        """

        def verdict(**fields: Any) -> Verdict:
            return Verdict(
                watchdog=defn.name,
                engine="decision",
                session_key=session_key,
                record_session_id=record_session_id,
                submission_id=submission_id,
                threshold=defn.flag_threshold,
                uncalibrated=True,
                **fields,
            )

        started = self._clock()
        digest = InputDigest()
        calls: list[JevCall] = []

        def unavailable(error_type: ErrorType) -> Verdict:
            return verdict(
                status="unavailable",
                error_type=error_type,
                input_digest=digest,
                latency_s=_latency(calls, self._clock() - started),
                cost_usd=_total_cost(calls),
            )

        try:
            client, state, question, digest = await asyncio.to_thread(
                self._prepare, defn, nodes, prompt
            )
            result = await asyncio.to_thread(
                client.decide, state, {QUESTION_ID: question}, budget=BUDGET, deadline=deadline
            )
        except DecisionError as exc:
            calls = list(exc.completed_calls)
            logger.warning(
                "Watchdog %s is unavailable: %s (status %s)",
                defn.name,
                type(exc).__name__,
                exc.status,
            )
            return unavailable(_error_type(exc))
        except Exception as exc:
            logger.warning("Watchdog %s is unavailable: %s", defn.name, type(exc).__name__)
            return unavailable("decision_error")

        calls = list(result.calls)
        answer = result.answers.get(QUESTION_ID)
        if not isinstance(answer, ChoiceAnswer):
            return unavailable("decision_error")
        if not answer.probabilities:
            return unavailable("no_probabilities")

        flag_keys = [
            (_option_key(index), option.text)
            for index, option in enumerate(defn.options)
            if option.flag
        ]
        flag_mass = [(answer.probabilities.get(key, 0.0), text) for key, text in flag_keys]
        score = round(sum(mass for mass, _ in flag_mass), PROBABILITY_DECIMALS)
        flagged = score >= defn.flag_threshold
        return verdict(
            status="ok",
            flagged=flagged,
            reason=_top_text(flag_mass) if flagged else None,
            score=score,
            input_digest=digest,
            latency_s=_latency(calls, self._clock() - started),
            cost_usd=_total_cost(calls),
        )

    def _prepare(
        self,
        defn: WatchdogDefinition,
        nodes: SessionNodes | Sequence[Mapping[str, Any]],
        prompt: str,
    ) -> tuple[JevClient, str, ChoiceQuestion, InputDigest]:
        """The client, the wire state, the question and the state's digest."""
        client = self._get_client()
        instructions = defn.instructions.strip()
        question = _build_question(defn, instructions)
        prefix = _state_prefix(instructions)
        limit = client.settings.active_route.max_state_plus_question_tokens
        budget = (
            limit - estimate_state_plus_longest("", {QUESTION_ID: question}) - _state_tokens(prefix)
        )
        read = as_session_nodes(nodes)
        view = build_session_view_from_nodes(
            read.nodes, prompt, max_tokens=max(budget, 0), count_tokens=_state_tokens
        )
        state = prefix + view.text
        digest = InputDigest(
            sha256=hashlib.sha256(state.encode("utf-8")).hexdigest(),
            entries_kept=view.entries_kept,
            entries_cut=view.entries_cut,
            tokens_estimate=_state_tokens(state),
            drain_timed_out=read.drain_timed_out,
            record_available=read.record_available,
        )
        return client, state, question, digest

    def _get_client(self) -> JevClient:
        with self._client_lock:
            if self._client is None:
                self._client = JevClient(
                    JevSettings.from_config(self._config),
                    self._config,
                    transport=self._transport,
                    clock=self._clock,
                    sleep=self._sleep,
                )
            return self._client


def _latency(calls: Sequence[JevCall], elapsed: float) -> float:
    return sum(call.latency_s for call in calls) if calls else elapsed


def _top_text(flag_mass: Sequence[tuple[float, str]]) -> str:
    """The text of the likeliest flag option; the earlier option wins a tie."""
    return max(flag_mass, key=lambda entry: entry[0])[1]
