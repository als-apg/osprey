"""Unit tests for the dispatch-worker failure-class taxonomy and stamping."""

from __future__ import annotations

import pytest

from osprey.agent_runner.verdict import _BUDGET_SUBTYPES
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


@pytest.fixture(autouse=True)
def _clear_counter_hook():
    """Ensure counter-hook state never leaks between tests."""
    fc.register_counter_hook(None)
    yield
    fc.register_counter_hook(None)


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
# Counter hook seam
# ---------------------------------------------------------------------------


def test_stamp_invokes_registered_counter_hook():
    seen: list[str] = []
    fc.register_counter_hook(seen.append)
    fc._stamp({"status": "error"}, fc.FAILURE_PROVIDER, 1)
    fc._stamp({"status": "error"}, fc.FAILURE_PROVIDER, 2)
    fc._stamp({"status": "error"}, fc.FAILURE_RUN, 0)
    assert seen == [fc.FAILURE_PROVIDER, fc.FAILURE_PROVIDER, fc.FAILURE_RUN]


def test_stamp_without_hook_is_a_noop_for_counting():
    # No hook registered (autouse fixture cleared it) -> stamping still works.
    result = fc._stamp({"status": "error"}, fc.FAILURE_RUN, None)
    assert result["failure_class"] == fc.FAILURE_RUN


def test_counter_hook_exception_does_not_break_stamping():
    def _boom(_cls: str) -> None:
        raise RuntimeError("counter backend down")

    fc.register_counter_hook(_boom)
    # The stamp must still succeed even though the hook raised.
    result = fc._stamp({"status": "error"}, fc.FAILURE_INFRASTRUCTURE, 5)
    assert result["failure_class"] == fc.FAILURE_INFRASTRUCTURE
    assert result["num_tool_calls"] == 5


def test_register_counter_hook_none_detaches():
    seen: list[str] = []
    fc.register_counter_hook(seen.append)
    fc._stamp({"status": "error"}, fc.FAILURE_RUN, 0)
    fc.register_counter_hook(None)
    fc._stamp({"status": "error"}, fc.FAILURE_RUN, 0)
    assert seen == [fc.FAILURE_RUN]  # only the first call was counted
