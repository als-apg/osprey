"""The judge asks for its verdict again when the reply cannot be read, extracts
values without seeing references, and compares reported values in code.

No model is reached: the provider call is replaced by a fake that answers on a
script, so what is pinned is the judge's own rule -- an unreadable reply is
asked again a bounded number of times, and any other failure of the call is
raised as it comes. Model-independent, so the benchmark matrix leaves this
file out of its per-model counts (``scripts/benchmark/matrix_e2e_config.json``).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tests.e2e import judge as judge_module
from tests.e2e.judge import (
    JUDGE_ATTEMPTS,
    UNPARSED_VERDICT,
    JudgeEvaluation,
    LLMJudge,
    Reference,
    check_reported,
)


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


class _Reported(BaseModel):
    beta: float | None = None


async def test_extraction_asks_for_the_callers_fields_and_returns_them(monkeypatch):
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return _Reported(beta=14.93)

    monkeypatch.setattr(judge_module, "get_chat_completion", fake)

    reported = await _judge().extract("beta_x at BPM01 is 14.93 m", _Reported, "- beta: beta")

    assert reported == _Reported(beta=14.93)
    assert seen["output_model"] is _Reported
    assert "beta_x at BPM01 is 14.93 m" in seen["message"]


async def test_an_unreadable_extraction_is_asked_again(monkeypatch):
    fake = _Scripted(_unparsed(), _Reported(beta=1.0))
    monkeypatch.setattr(judge_module, "get_chat_completion", fake)

    assert await _judge().extract("text", _Reported, "- beta: beta") == _Reported(beta=1.0)
    assert fake.calls == 2


def test_values_within_tolerance_report_nothing():
    assert (
        check_reported(
            {"circumference": 182.121951, "beta": 14.93},
            {
                "circumference": Reference(182.12195088, 1e-6, relative=True),
                "beta": Reference(14.9336, 0.01, relative=True),
            },
        )
        == []
    )


def test_a_relative_miss_names_the_value_the_reference_and_the_error():
    (failure,) = check_reported(
        {"circumference": 182.12}, {"circumference": Reference(182.12195088, 1e-6, relative=True)}
    )
    assert failure == (
        "circumference: reported 182.12, reference 182.12195088, "
        "relative error 1.07e-05 exceeds 1e-06"
    )


def test_a_missing_value_fails():
    assert check_reported({"beta": None}, {"beta": Reference(6.05, 0.01, relative=True)}) == [
        "beta: not reported (reference 6.05)"
    ]


@pytest.mark.parametrize("reported", [0.2207, 14.2207, 0.9999 + 0.2208])
def test_a_tune_compares_on_its_fractional_part(reported):
    assert check_reported({"tune": reported}, {"tune": Reference(0.22072, 1e-3, modulo=1.0)}) == []


def test_a_tune_across_the_integer_boundary_is_close():
    assert check_reported({"tune": 0.9995}, {"tune": Reference(0.0004, 1e-3, modulo=1.0)}) == []


def test_a_wrong_tune_fails_absolutely():
    (failure,) = check_reported({"tune": 0.25}, {"tune": Reference(0.22072, 1e-3, modulo=1.0)})
    assert "absolute error 2.93e-02 exceeds 1e-03" in failure


def test_a_nan_never_passes():
    assert check_reported({"beta": float("nan")}, {"beta": Reference(6.05, 0.01, relative=True)})
