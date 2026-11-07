"""A provider's refusal is not the model's failure (tests/e2e/provider_refusal.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.e2e import provider_refusal

UNSERVED = (
    "API Error: 400 anthropic_messages: Invalid model name passed in model=claude-sonnet-5-5. "
    "Call `/v1/models` to view available models for your key."
)
FORBIDDEN = "Failed to authenticate. API Error: 403 Access denied: IP address is not recognized."


@pytest.fixture(autouse=True)
def _on_the_board(monkeypatch):
    monkeypatch.setenv("OSPREY_E2E_LANES", "lanes.json")
    monkeypatch.setenv("OSPREY_E2E_FORCE_PROVIDER", "als-apg")
    monkeypatch.setenv("OSPREY_E2E_FORCE_MODEL", "m")
    provider_refusal.take()


@pytest.mark.parametrize(
    ("status", "text", "model_turns", "refused"),
    [
        (400, UNSERVED, False, True),
        (400, UNSERVED, True, True),
        (None, FORBIDDEN, False, True),
        (None, FORBIDDEN, True, True),
        (401, "Unauthorized", True, True),
        (404, "not found", True, True),
        (None, "API Error: Connection error.", True, True),
        (400, "API Error: 400 prompt is too long: 250000 tokens", True, False),
        (400, "API Error: 400 prompt is too long: 250000 tokens", False, False),
        (429, "rate limited", True, False),
        (429, "rate limited", False, True),
        (529, "overloaded", True, False),
        (500, "server error", False, True),
        (None, "Reached maximum number of turns", True, False),
        (None, None, False, False),
    ],
)
def test_the_rule(status, text, model_turns, refused):
    reason = provider_refusal.refusal_reason(status=status, text=text, model_turns=model_turns)
    assert (reason is not None) is refused


def _result(**fields) -> SimpleNamespace:
    defaults = {"is_error": True, "api_error_status": None, "result": None, "usage": None}
    return SimpleNamespace(**{**defaults, **fields})


def _report(when: str, outcome: str) -> SimpleNamespace:
    return SimpleNamespace(when=when, outcome=outcome, longrepr="the test's own failure")


def test_a_refusal_after_real_turns_makes_the_test_a_setup_error():
    result = _result(result=FORBIDDEN, usage={"output_tokens": 900})
    provider_refusal.note_session(result, tool_calls=4)

    report = _report("call", "failed")
    provider_refusal.apply_to_report(report)

    assert report.when == "setup"
    assert report.outcome == "failed"
    assert "als-apg refused m: 403" in report.longrepr


def test_a_test_that_passed_through_a_refused_session_is_not_a_pass():
    provider_refusal.note_session(_result(result=UNSERVED), tool_calls=0)
    report = _report("call", "passed")
    provider_refusal.apply_to_report(report)
    assert (report.when, report.outcome) == ("setup", "failed")


def test_a_rate_limit_after_real_turns_keeps_the_tests_outcome():
    result = _result(api_error_status=429, result="rate limited", usage={"output_tokens": 50})
    provider_refusal.note_session(result, tool_calls=2)
    report = _report("call", "failed")
    provider_refusal.apply_to_report(report)
    assert report.when == "call"


def test_a_session_that_did_not_error_notes_nothing():
    provider_refusal.note_session(_result(is_error=False, result=FORBIDDEN), tool_calls=0)
    assert provider_refusal.take() is None


def test_a_refusal_does_not_carry_over_to_the_next_test():
    provider_refusal.note_session(_result(result=FORBIDDEN), tool_calls=0)
    provider_refusal.apply_to_report(_report("setup", "passed"))
    assert provider_refusal.take() is None


def test_off_the_board_nothing_changes(monkeypatch):
    monkeypatch.delenv("OSPREY_E2E_LANES")
    provider_refusal.note_session(_result(result=FORBIDDEN), tool_calls=0)
    provider_refusal.note_judge_error(
        SimpleNamespace(status_code=403) and RuntimeError("API Error: 403"),
        provider="p",
        model="j",
    )
    assert provider_refusal.take() is None


def test_a_judge_call_the_provider_refused_is_noted():
    class Forbidden(Exception):
        status_code = 403

    provider_refusal.note_judge_error(Forbidden("denied"), provider="als-apg", model="j")
    assert "als-apg refused judge j: 403" in provider_refusal.take()


def test_a_judge_that_returned_something_unreadable_is_not_a_refusal():
    provider_refusal.note_judge_error(ValueError("bad json"), provider="als-apg", model="j")
    assert provider_refusal.take() is None
