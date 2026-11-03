"""Live Jev checks: owner-run only, never part of CI.

Each test spends a few cents against a real account. Run them by hand, with the
key exported into the shell that starts pytest (a value that only an ancestor
``.env`` supplies does not count, and the test skips)::

    export TYPESAFE_API_KEY=...        # direct route
    OSPREY_E2E_PROVIDER=als-apg uv run pytest tests/e2e/test_jev_live.py -m requires_typesafe -rs

    export OPENROUTER_API_KEY=...      # OpenRouter route
    OSPREY_E2E_PROVIDER=als-apg uv run pytest tests/e2e/test_jev_live.py -m requires_openrouter -rs

``OSPREY_E2E_PROVIDER`` is set only because the e2e conftest refuses to collect
without it; these tests make no call through that provider. With a key unset
the matching test skips.

A 401 or 402 means the key or the account balance is at fault: the test fails
once, naming only the error class, type and status, and must not be retried in
a loop. No test reads or prints a key, and no raw Jev error body is logged or
stored — a failure reports the error's type and status, never its message.

What the OpenRouter smoke test records (served model, whether ``probabilities``
and ``usage.cost`` came back, billed against list cost) is a finding for the
ship list. It is never turned into a default.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import JevSettings
from osprey.models.jev.errors import DecisionError
from osprey.models.jev.models import ChoiceAnswer, JevResult, NoulAnswer, ScoreAnswer
from osprey.models.providers.openrouter import OpenRouterProviderAdapter

STATE = "A water tank in a test plant reads 80 percent full and its level is steady."

NOUL = {"type": "noul", "instructions": "Is the tank more than half full?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which quantity does the state describe?",
    "criteria": {"level": "a tank fill level", "temp": "a temperature", "flow": "a flow rate"},
}
SCORE = {
    "type": "score",
    "instructions": "How full is the tank?",
    "criteria": ["empty", "about half full", "nearly full"],
}

# The direct route with every default; its key resolves from TYPESAFE_API_KEY.
TYPESAFE_CONFIG: dict[str, Any] = {"api": {"jev": {"route": "typesafe"}}}
# The OpenRouter route; with no api.providers.openrouter block the key resolves
# from OPENROUTER_API_KEY and the base from the provider's own default.
OPENROUTER_CONFIG: dict[str, Any] = {"api": {"jev": {"route": "openrouter"}}}

# pytest's ``record_property`` fixture.
RecordProperty = Callable[[str, object], None]


def _client(config: dict[str, Any]) -> Iterator[JevClient]:
    with JevClient(JevSettings.from_config(config), config) as client:
        yield client


@pytest.fixture
def typesafe_client() -> Iterator[JevClient]:
    yield from _client(TYPESAFE_CONFIG)


@pytest.fixture
def openrouter_client() -> Iterator[JevClient]:
    yield from _client(OPENROUTER_CONFIG)


def _decide(client: JevClient, questions: dict[str, Any]) -> JevResult:
    """One batch-budget decide; a failure names its class, type and status only."""
    try:
        return client.decide(STATE, questions, budget="batch")
    except DecisionError as err:
        stop = ""
        if err.status in (401, 402):
            stop = " (key or balance: stop and tell the owner, do not retry)"
        pytest.fail(
            f"Jev call failed: {type(err).__name__} error_type={err.error_type} "
            f"status={err.status}{stop}",
            pytrace=False,
        )


def _assert_noul(answer: Any) -> None:
    assert isinstance(answer, NoulAnswer)
    assert 0.0 <= answer.noul <= 1.0


def _assert_choice(answer: Any) -> None:
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice in CHOICE["criteria"]
    if answer.probabilities is not None:
        assert set(answer.probabilities) <= set(CHOICE["criteria"])
        assert all(0.0 <= p <= 1.0 for p in answer.probabilities.values())


def _assert_calls(result: JevResult) -> None:
    assert result.calls, "a completed decide records at least one call"
    for call in result.calls:
        assert call.input_tokens > 0
        assert call.output_tokens >= 0
        assert call.cost_usd >= 0.0
        assert call.list_cost_usd >= 0.0
        assert call.attempts >= 1


@pytest.mark.requires_typesafe
def test_direct_route_answers_one_question_of_each_type(
    typesafe_client: JevClient, record_property: RecordProperty
) -> None:
    """One noul, one choice and one score on the direct route; shapes and served model.

    Needs ``export TYPESAFE_API_KEY=...`` in the shell that runs pytest.
    """
    result = _decide(typesafe_client, {"q_noul": NOUL, "q_choice": CHOICE, "q_score": SCORE})

    assert set(result.answers) == {"q_noul", "q_choice", "q_score"}
    _assert_noul(result.answers["q_noul"])
    _assert_choice(result.answers["q_choice"])
    score = result.answers["q_score"]
    assert isinstance(score, ScoreAnswer)
    assert 0.0 <= score.score <= len(SCORE["criteria"]) - 1
    assert score.legend

    assert result.route == "typesafe"
    assert result.requested_model == typesafe_client.settings.routes.typesafe.model
    assert isinstance(result.served_model, str) and result.served_model.strip()
    _assert_calls(result)
    for call in result.calls:
        assert call.billed_cost_usd is None
        assert call.cost_source == "computed"

    record_property("served_model", result.served_model)
    print(f"\njev direct route: served_model={result.served_model}")


@pytest.mark.requires_openrouter
def test_openrouter_route_smoke(
    openrouter_client: JevClient, record_property: RecordProperty
) -> None:
    """One noul and one choice on the Jev OpenRouter route, one chat on the health model.

    Needs ``export OPENROUTER_API_KEY=...`` in the shell that runs pytest. Records
    the served model, whether ``probabilities`` and ``usage.cost`` came back, and
    the billed against the list cost; those are presence and shape checks only.
    """
    result = _decide(openrouter_client, {"q_noul": NOUL, "q_choice": CHOICE})

    assert set(result.answers) == {"q_noul", "q_choice"}
    _assert_noul(result.answers["q_noul"])
    choice = result.answers["q_choice"]
    _assert_choice(choice)
    assert isinstance(choice, ChoiceAnswer)
    assert result.route == "openrouter"
    assert result.requested_model == openrouter_client.settings.routes.openrouter.model
    _assert_calls(result)

    billed = [call.billed_cost_usd for call in result.calls]
    listed = [call.list_cost_usd for call in result.calls]
    findings = {
        "served_model": result.served_model,
        "choice_probabilities_present": choice.probabilities is not None,
        "usage_cost_present": all(value is not None for value in billed),
        "billed_cost_usd": None if None in billed else sum(b for b in billed if b is not None),
        "list_cost_usd": sum(listed),
    }
    for call in result.calls:
        expected = "billed" if call.billed_cost_usd is not None else "computed"
        assert call.cost_source == expected

    # One chat completion on the provider's health model; litellm reads
    # OPENROUTER_API_KEY from the environment itself, so no key passes through here.
    adapter = OpenRouterProviderAdapter()
    usage: list[dict[str, Any]] = []
    reply = adapter.execute_completion(
        message="Reply with the single word OK.",
        model_id=adapter.health_check_model_id,
        api_key=None,
        base_url=None,
        max_tokens=16,
        usage_sink=usage.append,
    )
    assert isinstance(reply, str) and reply.strip()
    assert usage and usage[0]["input_tokens"] > 0
    findings["chat_model"] = adapter.health_check_model_id

    for name, value in findings.items():
        record_property(name, value)
    print("\njev openrouter smoke: " + ", ".join(f"{k}={v}" for k, v in findings.items()))
