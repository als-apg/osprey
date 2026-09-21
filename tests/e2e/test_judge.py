"""The judge asks for its verdict again when the reply cannot be read.

No model is reached: the provider call is replaced by a fake that answers on a
script, so what is pinned is the judge's own rule -- an unreadable reply is
asked again a bounded number of times, and any other failure of the call is
raised as it comes. Model-independent, so the benchmark matrix leaves this
file out of its per-model counts (``scripts/benchmark/matrix_e2e_config.json``).
"""

from __future__ import annotations

import pytest

from tests.e2e import judge as judge_module
from tests.e2e.judge import JUDGE_ATTEMPTS, UNPARSED_VERDICT, JudgeEvaluation, LLMJudge


def _verdict() -> JudgeEvaluation:
    return JudgeEvaluation(passed=True, reasoning="the run did what was asked", confidence=0.9)


def _unparsed() -> ValueError:
    return ValueError(f"{UNPARSED_VERDICT} from als-apg: 1 validation error for JudgeEvaluation")


def _judge() -> LLMJudge:
    return LLMJudge(
        provider="als-apg",
        provider_config={"api_key": "unused", "base_url": "http://gateway.invalid"},
    )


class _Scripted:
    """Answers each call from a script: an exception is raised, anything else returned."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def test_a_readable_verdict_is_taken_on_the_first_ask(monkeypatch):
    fake = _Scripted(_verdict())
    monkeypatch.setattr(judge_module, "get_chat_completion", fake)

    verdict = _judge()._verdict("prompt")

    assert verdict.passed is True
    assert fake.calls == 1


def test_an_unreadable_verdict_is_asked_again_until_one_reads(monkeypatch):
    fake = _Scripted(_unparsed(), _unparsed(), _verdict())
    monkeypatch.setattr(judge_module, "get_chat_completion", fake)

    verdict = _judge()._verdict("prompt")

    assert verdict.passed is True
    assert fake.calls == JUDGE_ATTEMPTS


def test_the_last_unreadable_verdict_is_reported(monkeypatch):
    fake = _Scripted(*[_unparsed() for _ in range(JUDGE_ATTEMPTS)])
    monkeypatch.setattr(judge_module, "get_chat_completion", fake)

    with pytest.raises(ValueError, match=UNPARSED_VERDICT):
        _judge()._verdict("prompt")

    assert fake.calls == JUDGE_ATTEMPTS


def test_any_other_failure_is_raised_at_once(monkeypatch):
    fake = _Scripted(ValueError("Provider 'als-apg' refused the request"), _verdict())
    monkeypatch.setattr(judge_module, "get_chat_completion", fake)

    with pytest.raises(ValueError, match="refused"):
        _judge()._verdict("prompt")

    assert fake.calls == 1
