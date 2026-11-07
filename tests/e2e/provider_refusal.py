"""A session the provider refused says nothing about the model under test.

The benchmark board records such a test as a setup error, which leaves the score
and the median task time, by the rule below:

* Refused at any point, after real turns too: auth and billing (401, 402, 403), a
  model the provider does not serve (404, or a 400 that says so), and an endpoint
  that cannot be reached. Retrying changes none of these, and none is the model's.
* A rate limit or a 5xx (429, 500-599): the provider's refusal only when no model
  turn ever happened. After real turns the test keeps its outcome, because the
  retries are spent and throughput under load is part of what a run measures.
* Any other 400, such as a request past the model's context window, stays the
  test's outcome: the model's own limits are what the board compares.

Off outside the board (no ``OSPREY_E2E_LANES``), so a CI lane fails as it always did.
"""

import os
import re
from typing import Any

#: Refused at any point in a session.
_ALWAYS_REFUSED = frozenset({401, 402, 403, 404})
#: A 400 that names an unserved model; any other 400 is not a refusal.
_UNSERVED_MODEL = re.compile(
    r"invalid model|model[_ ]not[_ ]found|unknown model|model .*(does not exist|not found)"
    r"|not a valid model",
    re.IGNORECASE,
)
#: What the agent CLI reports when no response came back at all.
_UNREACHABLE = re.compile(
    r"connection error|unable to connect|econnrefused|enotfound|fetch failed", re.IGNORECASE
)
_API_ERROR_STATUS = re.compile(r"API Error: (\d{3})")

_noted: list[str] = []


def refusal_reason(*, status: int | None, text: str | None, model_turns: bool) -> str | None:
    """Why this failed call was the provider's refusal, or None when it was not."""
    text = text or ""
    if status is None:
        found = _API_ERROR_STATUS.search(text)
        status = int(found.group(1)) if found else None
    if status is None:
        return f"unreachable: {text[:200]}" if _UNREACHABLE.search(text) else None
    refused = (
        status in _ALWAYS_REFUSED
        or (status == 400 and _UNSERVED_MODEL.search(text) is not None)
        or ((status == 429 or 500 <= status <= 599) and not model_turns)
    )
    return f"{status}: {text[:200]}" if refused else None


def note_session(result: Any, *, tool_calls: int) -> None:
    """Note a finished agent session the provider refused, on the board only."""
    if not os.environ.get("OSPREY_E2E_LANES") or result is None or not result.is_error:
        return
    usage = getattr(result, "usage", None) or {}
    reason = refusal_reason(
        status=getattr(result, "api_error_status", None),
        text=getattr(result, "result", None),
        model_turns=bool(usage.get("output_tokens")) or tool_calls > 0,
    )
    if reason:
        _noted.append(
            f"{os.environ.get('OSPREY_E2E_FORCE_PROVIDER', 'provider')} refused "
            f"{os.environ.get('OSPREY_E2E_FORCE_MODEL', 'the model')}: {reason}"
        )


def note_judge_error(error: BaseException, *, provider: str, model: str) -> None:
    """Note a judge call the provider refused, on the board only."""
    if not os.environ.get("OSPREY_E2E_LANES"):
        return
    status = getattr(error, "status_code", None)
    text = str(error)
    if status is None and "connection" in type(error).__name__.lower():
        text = f"connection error: {text}"
    reason = refusal_reason(status=status, text=text, model_turns=False)
    if reason:
        _noted.append(f"{provider} refused judge {model}: {reason}")


def take() -> str | None:
    """The refusal noted since the last call, if any, and forget it."""
    reason = _noted[0] if _noted else None
    _noted.clear()
    return reason


def apply_to_report(report: Any) -> None:
    """Make a refused test's call report a setup error, everywhere it is read.

    Rewriting ``when`` is what every reader agrees on: pytest's terminal and JUnit
    writers report a failure outside the call as an error, and so does the live
    stream. A test that passed despite the refusal is rewritten too, since the
    session it graded did not finish.
    """
    if report.when == "setup":
        take()
    elif report.when == "call":
        reason = take()
        if reason and report.outcome in ("passed", "failed"):
            report.outcome = "failed"
            report.when = "setup"
            report.longrepr = f"provider refusal: {reason}"
