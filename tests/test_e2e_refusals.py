"""A provider's refusal is not the model's failure (tests/e2e/refusals.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.e2e import refusals

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
    refusals.take()


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
    reason = refusals.refusal_reason(status=status, text=text, model_turns=model_turns)
    assert (reason is not None) is refused


def _result(**fields) -> SimpleNamespace:
    defaults = {"is_error": True, "api_error_status": None, "result": None, "usage": None}
    return SimpleNamespace(**{**defaults, **fields})


def _report(when: str, outcome: str) -> SimpleNamespace:
    return SimpleNamespace(when=when, outcome=outcome, longrepr="the test's own failure")


def test_a_refusal_after_real_turns_makes_the_test_a_setup_error():
    result = _result(result=FORBIDDEN, usage={"output_tokens": 900})
    refusals.note_session(result, tool_calls=4)

    report = _report("call", "failed")
    refusals.apply_to_report(report)

    assert report.when == "setup"
    assert report.outcome == "failed"
    assert "als-apg refused m: 403" in report.longrepr


def test_a_test_that_passed_through_a_refused_session_is_not_a_pass():
    refusals.note_session(_result(result=UNSERVED), tool_calls=0)
    report = _report("call", "passed")
    refusals.apply_to_report(report)
    assert (report.when, report.outcome) == ("setup", "failed")


def test_a_rate_limit_after_real_turns_keeps_the_tests_outcome():
    result = _result(api_error_status=429, result="rate limited", usage={"output_tokens": 50})
    refusals.note_session(result, tool_calls=2)
    report = _report("call", "failed")
    refusals.apply_to_report(report)
    assert report.when == "call"


def test_a_session_that_did_not_error_notes_nothing():
    refusals.note_session(_result(is_error=False, result=FORBIDDEN), tool_calls=0)
    assert refusals.take() is None


def test_a_refusal_does_not_carry_over_to_the_next_test():
    refusals.note_session(_result(result=FORBIDDEN), tool_calls=0)
    refusals.apply_to_report(_report("setup", "passed"))
    assert refusals.take() is None


def test_off_the_board_nothing_changes(monkeypatch):
    monkeypatch.delenv("OSPREY_E2E_LANES")
    refusals.note_session(_result(result=FORBIDDEN), tool_calls=0)
    refusals.note_judge_error(
        SimpleNamespace(status_code=403) and RuntimeError("API Error: 403"),
        provider="p",
        model="j",
    )
    assert refusals.take() is None


def test_a_judge_call_the_provider_refused_is_noted():
    class Forbidden(Exception):
        status_code = 403

    refusals.note_judge_error(Forbidden("denied"), provider="als-apg", model="j")
    assert "als-apg refused judge j: 403" in refusals.take()


def test_a_judge_that_returned_something_unreadable_is_not_a_refusal():
    refusals.note_judge_error(ValueError("bad json"), provider="als-apg", model="j")
    assert refusals.take() is None


@pytest.mark.parametrize(
    ("text", "kept", "gone"),
    [
        ("Access denied: IP address 2a01:4f8::1 is not recognized.", "Access denied", "2a01"),
        ("from fe80::1%en0.", "from <address>", "fe80"),
        ("loopback ::1 refused", "loopback <address> refused", "::1"),
        ("mapped ::ffff:a00:1 refused", "<address>", "ffff"),
        ("IP address 10.1.2.3 is not recognized", "<address>", "10.1.2.3"),
        (
            "Authentication Error, Invalid proxy server token passed. Received API Key = "
            "sk-...a7Qz, Key Hash (Token) =fb3c0a8e1d2f4b5a6c7d8e9f0a1b2c3d4e5f60718293a4b5c6d7e8f9a0"
            "b1c2d3. Unable to find token in cache",
            "Authentication Error, Invalid proxy server token passed.",
            "a7Qz",
        ),
        ("401 Bearer abc.def denied", "401 Bearer *** denied", "abc.def"),
        ("see https://api.gateway.example/key/manage now", "see <url> now", "gateway"),
        ("gateway gw.example.org said no", "gateway <host> said no", "example"),
    ],
)
def test_the_scrubber_keeps_the_reason_and_drops_what_identifies(text, kept, gone):
    scrubbed = refusals.scrub(text)
    assert kept in scrubbed
    assert gone not in scrubbed


@pytest.mark.parametrize(
    "text", ["at 12:30:45", "model qwen2.5:32b", "claude-sonnet-5-5", "gpt-4.1"]
)
def test_the_scrubber_leaves_times_and_model_names_alone(text):
    assert refusals.scrub(text) == text


def test_the_scrubber_drops_the_exact_key():
    assert refusals.scrub("bad key abc123xyz", key="abc123xyz") == "bad key ***"


def test_a_refusal_after_real_turns_is_scrubbed_before_it_reaches_the_report():
    text = "API Error: 403 Access denied: IP address 2001:5a8::36b8 not recognized, sk-live-9f2c"
    refusals.note_session(_result(result=text, usage={"output_tokens": 300}), tool_calls=3)
    report = _report("call", "failed")
    refusals.apply_to_report(report)
    assert "2001:5a8" not in report.longrepr and "sk-live" not in report.longrepr
    assert "403" in report.longrepr and "<address>" in report.longrepr
