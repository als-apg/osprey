"""Table-driven coverage of the pure retry-policy decisions in osprey.bridges.core.retry.

Each branch of every function gets an explicit row: retryable failure classes,
the redispatch safety gate (known tool-call counts), age eligibility, the
two-ceiling give-up rule, and coalesce-key normalization."""

from types import SimpleNamespace

import pytest

from osprey.bridges.core.retry import (
    coalesce_key,
    give_up_due,
    is_eligible,
    is_retryable,
    may_redispatch,
)

NOW = 1_000_000.0


def _cfg(retry_min_age=1200.0, retry_give_up=172800.0, retry_lifetime_cap=604800.0):
    return SimpleNamespace(
        retry_min_age=retry_min_age,
        retry_give_up=retry_give_up,
        retry_lifetime_cap=retry_lifetime_cap,
    )


def _queued(age, **extra):
    """An entry that entered the queue ``age`` seconds before NOW."""
    return {"queued_at": NOW - age, **extra}


# --- is_retryable ----------------------------------------------------------


@pytest.mark.parametrize(
    "failure_class, expected",
    [
        ("provider", True),
        ("infrastructure", True),
        ("agent", False),
        ("user", False),
        ("", False),
        (None, False),
    ],
)
def test_is_retryable_by_class(failure_class, expected):
    assert is_retryable({"failure_class": failure_class}) is expected


def test_is_retryable_missing_key():
    assert is_retryable({}) is False


# --- is_retryable: error_code override -------------------------------------


@pytest.mark.parametrize(
    "error_code, failure_class, expected",
    [
        # The two input_files rejections are non-retryable even though the
        # bridge-local error result defaults failure_class to the (retryable)
        # "infrastructure" class -> the error_code must win.
        ("input_files_cap_exceeded", "infrastructure", False),
        ("input_files_invalid", "infrastructure", False),
        ("input_files_cap_exceeded", "provider", False),
        # An absent / unrelated error_code leaves the failure_class verdict intact.
        (None, "infrastructure", True),
        ("some_other_code", "provider", True),
        (None, "agent", False),
    ],
)
def test_is_retryable_error_code_override(error_code, failure_class, expected):
    assert is_retryable({"failure_class": failure_class, "error_code": error_code}) is expected


def test_is_retryable_error_code_absent_key_uses_failure_class():
    # No error_code key at all -> fall through to the failure_class rule.
    assert is_retryable({"failure_class": "provider"}) is True
    assert is_retryable({"failure_class": "agent"}) is False


# --- may_redispatch --------------------------------------------------------


@pytest.mark.parametrize(
    "result_n, entry_n, expected",
    [
        (0, None, True),  # result knows 0, entry silent
        (0, 0, True),  # both known 0
        (None, 0, True),  # entry remembers 0 from a prior attempt
        (0, 2, False),  # a prior attempt DID make tool calls -> unsafe
        (3, None, False),  # this run made tool calls
        (None, None, False),  # unknown everywhere -> fail-closed
        (True, None, False),  # bool is not a valid count -> unknown
    ],
)
def test_may_redispatch(result_n, entry_n, expected):
    entry = {} if entry_n is None else {"num_tool_calls": entry_n}
    result = {} if result_n is None else {"num_tool_calls": result_n}
    assert may_redispatch(entry, result) is expected


@pytest.mark.parametrize("n", [1, 2, 3, 7, 42, 1000])
def test_may_redispatch_refuses_whenever_tool_calls_were_made(n):
    """A run that made ANY tool call may never be replayed.

    A dispatched run can reach hardware, so a redispatch of a run with side
    effects repeats a machine action. The refusal must hold no matter which
    side reports the positive count, and no matter what the other side says.
    """
    # Positive count observed on the fresh result.
    assert may_redispatch({}, {"num_tool_calls": n}) is False
    # Positive count remembered on the entry from a prior attempt.
    assert may_redispatch({"num_tool_calls": n}, {}) is False
    # A known-0 on the other side does NOT rescue it — every known count must be 0.
    assert may_redispatch({"num_tool_calls": 0}, {"num_tool_calls": n}) is False
    assert may_redispatch({"num_tool_calls": n}, {"num_tool_calls": 0}) is False


# --- is_eligible -----------------------------------------------------------


@pytest.mark.parametrize(
    "age, expected",
    [
        (1201.0, True),  # just past min
        (1200.0, False),  # exactly min -> strict > fails
        (600.0, False),  # too young
        (-50.0, False),  # future queued_at (clock skew) -> age clamps to 0
    ],
)
def test_is_eligible_by_age(age, expected):
    assert is_eligible(_queued(age), NOW, _cfg()) is expected


def test_is_eligible_missing_queued_at():
    # No queued_at -> treated as just-now (age 0) -> not yet eligible.
    assert is_eligible({}, NOW, _cfg()) is False


# --- give_up_due -----------------------------------------------------------


@pytest.mark.parametrize(
    "age, attempts, gate_ever_open, expected",
    [
        # Hard lifetime cap: unconditional, even with no attempt and gate never open.
        (604800.0, 0, False, True),
        (700000.0, 0, False, True),
        # Give-up age reached AND the entry was actually retried.
        (172800.0, 1, False, True),
        # Give-up age reached AND the gate was seen open since enqueue.
        (172800.0, 0, True, True),
        # Give-up age reached but never retried and gate never opened -> extend.
        (172800.0, 0, False, False),
        (500000.0, 0, False, False),
        # Below the give-up age -> not due regardless of attempts/gate.
        (100000.0, 5, True, False),
    ],
)
def test_give_up_due(age, attempts, gate_ever_open, expected):
    entry = _queued(age, attempts=attempts)
    assert give_up_due(entry, NOW, gate_ever_open, _cfg()) is expected


def test_give_up_due_retried_flag_triggers():
    # The drain stamps a boolean `retried` flag (not a count) on redispatch;
    # give_up_due honors it the same as attempts > 0.
    entry = _queued(172800.0, retried=True)
    assert give_up_due(entry, NOW, False, _cfg()) is True


def test_give_up_due_retried_flag_false_extends():
    entry = _queued(172800.0, retried=False)
    assert give_up_due(entry, NOW, False, _cfg()) is False


def test_give_up_due_missing_queued_at_never_gives_up():
    # Age 0 -> below both ceilings.
    assert give_up_due({"attempts": 3}, NOW, True, _cfg()) is False


# --- ceiling anchor: first_queued_at ---------------------------------------


def test_give_up_due_anchors_on_first_queued_at():
    # A re-parked entry has a FRESH queued_at (per-attempt pacing) but its
    # first_queued_at remembers when the request originally entered the queue.
    # The give-up ceiling must read the original time, else a cycling entry
    # (park -> retry -> fail -> re-park) re-ages forever and never gives up.
    entry = {"queued_at": NOW - 100.0, "first_queued_at": NOW - 172800.0, "retried": True}
    assert give_up_due(entry, NOW, False, _cfg()) is True


def test_lifetime_cap_anchors_on_first_queued_at():
    entry = {"queued_at": NOW - 100.0, "first_queued_at": NOW - 604800.0}
    assert give_up_due(entry, NOW, False, _cfg()) is True


def test_give_up_due_falls_back_to_queued_at():
    # Entries persisted before first_queued_at existed keep working: the
    # ceilings fall back to queued_at.
    entry = {"queued_at": NOW - 604800.0}
    assert give_up_due(entry, NOW, False, _cfg()) is True


def test_is_eligible_ignores_first_queued_at():
    # Pacing stays per-park: a just-re-parked entry (fresh queued_at) is NOT
    # eligible again immediately, however old the request itself is.
    entry = {"queued_at": NOW - 100.0, "first_queued_at": NOW - 604800.0}
    assert is_eligible(entry, NOW, _cfg()) is False


# --- coalesce_key ----------------------------------------------------------


@pytest.mark.parametrize(
    "key, expected",
    [
        ("space/AAA", ("space/AAA",)),
        (["space", "thread"], ("space", "thread")),
        (("space", "thread"), ("space", "thread")),
        (None, ()),
        ("", ()),
        ([], ()),
        ((), ()),
    ],
)
def test_coalesce_key(key, expected):
    entry = {} if key is None else {"coalesce_key": key}
    assert coalesce_key(entry) == expected


def test_coalesce_key_missing():
    assert coalesce_key({}) == ()
