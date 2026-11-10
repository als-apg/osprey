"""Unit tests for the dispatch-worker failure-class taxonomy and stamping.

The three contracts the module guarantees to its stamp sites:
  - ``classify_exception`` maps an exception to ``provider`` via the type-name
    table, then the message-substring table, then defaults to ``run``, walking
    the cause/context chain bounded and cycle-safe, and never returning
    ``infrastructure``;
  - ``is_budget_subtype`` reflects the verdict module's budget subtypes rather
    than a second copy of them;
  - ``_stamp`` writes the class and tool-call count in place, rejects an
    unknown class without mutating, and bumps the per-class lifetime counter
    without letting a broken counter escape.
"""

from __future__ import annotations

import pytest

from osprey.agent_runner.verdict import _BUDGET_SUBTYPES
from osprey.mcp_server.dispatch_worker import counters
from osprey.mcp_server.dispatch_worker import failure_class as fc

# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------


def test_class_constants_are_the_three_expected_values():
    assert fc.FAILURE_PROVIDER == "provider"
    assert fc.FAILURE_INFRASTRUCTURE == "infrastructure"
    assert fc.FAILURE_RUN == "run"
    assert fc.FAILURE_CLASSES == {"provider", "infrastructure", "run"}


# ---------------------------------------------------------------------------
# classify_exception — provider by type name
# ---------------------------------------------------------------------------


def _make_exc(type_name: str, message: str = "") -> BaseException:
    """Build a throwaway exception whose class name is ``type_name``."""
    exc_cls = type(type_name, (Exception,), {})
    return exc_cls(message)


@pytest.mark.parametrize(
    "type_name",
    [
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "APIError",
        "APIStatusError",
        "APIConnectionError",
        "APITimeoutError",
        "APIResponseValidationError",
        "InternalServerError",
        "OverloadedError",
    ],
)
def test_provider_exception_type_names_map_to_provider(type_name):
    assert fc.classify_exception(_make_exc(type_name)) == fc.FAILURE_PROVIDER


# ---------------------------------------------------------------------------
# classify_exception — provider by message substring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - rate limit exceeded",
        "The upstream provider is Overloaded",
        "HTTP 529 from provider",
        "authentication_error: invalid x-api-key",
        "401 Unauthorized",
        "403 Forbidden",
        "Invalid API key provided",
        "expired credential",
        "permission denied for this resource",
        "insufficient_quota for this key",
    ],
)
def test_provider_message_substrings_map_to_provider(message):
    # A plain Exception whose *type* is not a known provider class still gets
    # classified by its message.
    assert fc.classify_exception(ValueError(message)) == fc.FAILURE_PROVIDER


def test_message_match_is_case_insensitive():
    assert fc.classify_exception(RuntimeError("RATE LIMIT hit")) == fc.FAILURE_PROVIDER


# ---------------------------------------------------------------------------
# classify_exception — run (default) and cause-chain walking
# ---------------------------------------------------------------------------


def test_unknown_exception_defaults_to_run():
    assert fc.classify_exception(ValueError("something went wrong in the agent")) == fc.FAILURE_RUN


def test_generic_runtime_error_defaults_to_run():
    assert fc.classify_exception(RuntimeError("tool raised")) == fc.FAILURE_RUN


def test_generic_classification_never_returns_infrastructure():
    # Worker-machinery faults are known at their call sites and stamped there,
    # so nothing routed through the generic mapping may claim to be one —
    # whichever of the other two classes it resolves to.
    for exc in (
        ValueError("agent produced bad output"),
        RuntimeError("rate limit"),
        _make_exc("RateLimitError", "429"),
    ):
        assert fc.classify_exception(exc) != fc.FAILURE_INFRASTRUCTURE


def test_provider_cause_wrapped_in_generic_is_detected():
    root = _make_exc("RateLimitError", "429")
    try:
        try:
            raise root
        except BaseException as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as wrapper:
        assert fc.classify_exception(wrapper) == fc.FAILURE_PROVIDER


def test_provider_context_chain_is_detected():
    # Implicit chaining via __context__ (no explicit ``from``).
    try:
        try:
            raise _make_exc("AuthenticationError", "bad key")
        except BaseException:
            raise RuntimeError("later failure")  # noqa: B904 - exercises implicit __context__
    except RuntimeError as wrapper:
        assert fc.classify_exception(wrapper) == fc.FAILURE_PROVIDER


def test_cause_chain_walk_is_cycle_safe():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # deliberate cycle
    # Must terminate and classify without hanging.
    assert fc.classify_exception(a) == fc.FAILURE_RUN


# ---------------------------------------------------------------------------
# is_budget_subtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subtype", sorted(_BUDGET_SUBTYPES))
def test_budget_subtypes_are_recognised(subtype):
    assert fc.is_budget_subtype(subtype) is True


@pytest.mark.parametrize("subtype", [None, "", "success", "error_during_execution"])
def test_non_budget_subtypes_are_not_recognised(subtype):
    assert fc.is_budget_subtype(subtype) is False


def test_is_budget_subtype_reuses_verdict_constant():
    # Guard against duplication: the module must not redefine its own copy.
    assert fc._BUDGET_SUBTYPES is _BUDGET_SUBTYPES


# ---------------------------------------------------------------------------
# _stamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure_class",
    [fc.FAILURE_PROVIDER, fc.FAILURE_INFRASTRUCTURE, fc.FAILURE_RUN],
)
def test_stamp_writes_class_and_tool_count(failure_class):
    result: dict = {"status": "error", "error": "boom"}
    returned = fc._stamp(result, failure_class, 3)
    assert returned is result  # mutates in place and returns the same dict
    assert result["failure_class"] == failure_class
    assert result["num_tool_calls"] == 3


def test_stamp_preserves_none_tool_count():
    result: dict = {"status": "error"}
    fc._stamp(result, fc.FAILURE_INFRASTRUCTURE, None)
    assert result["num_tool_calls"] is None


def test_stamp_rejects_unknown_class():
    with pytest.raises(ValueError, match="unknown failure_class"):
        fc._stamp({"status": "error"}, "bogus", 0)


def test_stamp_does_not_mutate_on_bad_class():
    result: dict = {"status": "error"}
    with pytest.raises(ValueError):
        fc._stamp(result, "nope", 1)
    assert "failure_class" not in result
    assert "num_tool_calls" not in result


# ---------------------------------------------------------------------------
# Per-class counter
# ---------------------------------------------------------------------------


def test_stamp_bumps_the_counter_per_class(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(counters, "increment", seen.append)
    fc._stamp({"status": "error"}, fc.FAILURE_PROVIDER, 1)
    fc._stamp({"status": "error"}, fc.FAILURE_PROVIDER, 2)
    fc._stamp({"status": "error"}, fc.FAILURE_RUN, 0)
    assert seen == [fc.FAILURE_PROVIDER, fc.FAILURE_PROVIDER, fc.FAILURE_RUN]


def test_stamp_counts_into_the_lifetime_counters():
    counters.reset()
    result = fc._stamp({"status": "error"}, fc.FAILURE_RUN, None)
    assert result["failure_class"] == fc.FAILURE_RUN
    assert counters.get_counts()[fc.FAILURE_RUN] == 1
    counters.reset()


def test_counter_exception_does_not_break_stamping(monkeypatch):
    def _boom(_cls: str) -> None:
        raise RuntimeError("counter backend down")

    monkeypatch.setattr(counters, "increment", _boom)
    # The stamp must still succeed even though the counter raised.
    result = fc._stamp({"status": "error"}, fc.FAILURE_INFRASTRUCTURE, 5)
    assert result["failure_class"] == fc.FAILURE_INFRASTRUCTURE
    assert result["num_tool_calls"] == 5
