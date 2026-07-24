"""Tests for the dispatch-worker failure-class taxonomy and stamping.

Covers the three contracts the module guarantees to its stamp sites:
  - ``classify_exception`` maps generic errors to exactly one class via the
    type-name table, then the message-substring table, then a ``run`` default,
    walking the cause/context chain (bounded, cycle-safe).
  - ``is_budget_subtype`` reflects the single-source-of-truth budget subtypes.
  - ``_stamp`` writes the class + tool-call count, rejects unknown classes, and
    bumps the registered counter hook without letting a broken hook escape.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.dispatch_worker import failure_class as fc


@pytest.fixture(autouse=True)
def _detach_hook():
    """Ensure no counter hook leaks across tests (module-global seam)."""
    yield
    fc.register_counter_hook(None)


# ---------------------------------------------------------------------------
# classify_exception
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Stand-in whose *name* matches the provider type table."""


class OverloadedError(Exception):
    pass


@pytest.mark.unit
def test_classify_by_type_name_is_provider():
    """A type name in the provider table classifies as provider, ignoring message."""
    assert fc.classify_exception(RateLimitError("anything at all")) == fc.FAILURE_PROVIDER
    assert fc.classify_exception(OverloadedError("boom")) == fc.FAILURE_PROVIDER


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        "Request hit a rate limit, retry later",
        "HTTP 429 Too Many Requests",
        "server is Overloaded",
        "401 Unauthorized",
        "invalid api key provided",
        "permission denied for this resource",
    ],
)
def test_classify_by_message_substring_is_provider(message):
    """A generic exception whose message carries a provider substring is provider."""
    assert fc.classify_exception(RuntimeError(message)) == fc.FAILURE_PROVIDER


@pytest.mark.unit
def test_classify_defaults_to_run():
    """An error with no provider signal is attributed to the run itself."""
    assert fc.classify_exception(ValueError("agent produced bad output")) == fc.FAILURE_RUN


@pytest.mark.unit
def test_classify_never_returns_infrastructure():
    """Generic mapping must never emit infrastructure (those sites stamp literally)."""
    for exc in (ValueError("x"), RuntimeError("rate limit"), RateLimitError("y")):
        assert fc.classify_exception(exc) != fc.FAILURE_INFRASTRUCTURE


@pytest.mark.unit
def test_classify_walks_cause_chain():
    """A provider error wrapped in a generic exception is still classified provider."""
    try:
        try:
            raise RateLimitError("upstream 429")
        except RateLimitError as inner:
            raise RuntimeError("wrapped and re-raised") from inner
    except RuntimeError as outer:
        assert fc.classify_exception(outer) == fc.FAILURE_PROVIDER


@pytest.mark.unit
def test_classify_type_name_beats_message_ordering():
    """Type-name match wins even when a shallower link only has a run-ish message."""
    # Outer has no provider signal in its message; inner cause is a provider type.
    inner = RateLimitError("")  # empty message => no substring signal, only the type
    outer = RuntimeError("agent step failed")
    outer.__cause__ = inner
    assert fc.classify_exception(outer) == fc.FAILURE_PROVIDER


@pytest.mark.unit
def test_classify_bounded_on_cyclic_chain():
    """A self-referential cause chain terminates instead of looping forever."""
    a = RuntimeError("plain agent error")
    b = RuntimeError("also plain")
    a.__cause__ = b
    b.__cause__ = a  # cycle
    assert fc.classify_exception(a) == fc.FAILURE_RUN


# ---------------------------------------------------------------------------
# is_budget_subtype
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("subtype", ["error_max_turns", "error_max_budget_usd"])
def test_is_budget_subtype_true(subtype):
    assert fc.is_budget_subtype(subtype) is True


@pytest.mark.unit
@pytest.mark.parametrize("subtype", ["success", "error_during_execution", "", None])
def test_is_budget_subtype_false(subtype):
    assert fc.is_budget_subtype(subtype) is False


# ---------------------------------------------------------------------------
# _stamp + counter seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stamp_writes_class_and_count():
    """_stamp annotates the result in place and returns the same dict."""
    result: dict = {"error": True}
    out = fc._stamp(result, fc.FAILURE_PROVIDER, num_tool_calls=3)
    assert out is result
    assert result["failure_class"] == fc.FAILURE_PROVIDER
    assert result["num_tool_calls"] == 3


@pytest.mark.unit
def test_stamp_accepts_none_tool_count():
    """None is a valid count (unknown at some sites) and is written through."""
    result = fc._stamp({}, fc.FAILURE_RUN, num_tool_calls=None)
    assert result["num_tool_calls"] is None


@pytest.mark.unit
def test_stamp_rejects_unknown_class():
    """An unrecognised class is a programming error, not a silent pass-through."""
    with pytest.raises(ValueError, match="unknown failure_class"):
        fc._stamp({}, "not_a_class", num_tool_calls=0)


@pytest.mark.unit
def test_stamp_bumps_registered_hook():
    """Every stamp invokes the counter hook with the class it stamped."""
    seen: list[str] = []
    fc.register_counter_hook(seen.append)
    fc._stamp({}, fc.FAILURE_INFRASTRUCTURE, num_tool_calls=0)
    fc._stamp({}, fc.FAILURE_PROVIDER, num_tool_calls=1)
    assert seen == [fc.FAILURE_INFRASTRUCTURE, fc.FAILURE_PROVIDER]


@pytest.mark.unit
def test_stamp_survives_broken_hook():
    """A raising hook is observability-only and must not fail the stamp."""

    def boom(_cls: str) -> None:
        raise RuntimeError("counter backend down")

    fc.register_counter_hook(boom)
    # Should not raise despite the hook blowing up.
    result = fc._stamp({}, fc.FAILURE_RUN, num_tool_calls=2)
    assert result["failure_class"] == fc.FAILURE_RUN


@pytest.mark.unit
def test_stamp_without_hook_is_noop_count():
    """With no hook registered, stamping still succeeds (counting is optional)."""
    fc.register_counter_hook(None)
    result = fc._stamp({}, fc.FAILURE_PROVIDER, num_tool_calls=0)
    assert result["failure_class"] == fc.FAILURE_PROVIDER
