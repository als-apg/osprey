"""Tests for the auth sidecar's per-user login attempt throttle."""

from __future__ import annotations

import pytest

from osprey.services.auth_sidecar.throttle import AttemptThrottle


class FakeClock:
    """A hand-advanced stand-in for ``time.monotonic``."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def throttle(clock: FakeClock) -> AttemptThrottle:
    return AttemptThrottle(clock=clock)


def test_first_attempt_is_allowed(throttle: AttemptThrottle) -> None:
    assert throttle.retry_after("alice") == 0.0


def test_failure_opens_the_initial_window(throttle: AttemptThrottle) -> None:
    assert throttle.record_failure("alice") == 1.0
    assert throttle.retry_after("alice") == pytest.approx(1.0)


def test_window_lifts_once_it_elapses(throttle: AttemptThrottle, clock: FakeClock) -> None:
    throttle.record_failure("alice")

    clock.advance(1.0)

    assert throttle.retry_after("alice") == 0.0


def test_window_still_holds_just_before_it_lifts(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    throttle.record_failure("alice")

    clock.advance(0.75)

    assert throttle.retry_after("alice") == pytest.approx(0.25)


def test_window_doubles_on_each_further_failure(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    windows = []
    for _ in range(4):
        windows.append(throttle.record_failure("alice"))
        clock.advance(windows[-1])

    assert windows == [1.0, 2.0, 4.0, 8.0]


def test_window_is_capped_at_thirty_seconds(throttle: AttemptThrottle, clock: FakeClock) -> None:
    """The cap is what bounds guessing throughput without ever locking anyone out."""
    for _ in range(20):
        delay = throttle.record_failure("alice")
        clock.advance(delay)

    assert delay == 30.0
    assert throttle.max_delay == 30.0


def test_window_never_exceeds_the_cap_however_many_failures(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    for _ in range(50):
        clock.advance(throttle.record_failure("alice"))

    assert throttle.retry_after("alice") == 0.0
    assert throttle.record_failure("alice") == 30.0
    assert throttle.retry_after("alice") <= 30.0


def test_success_resets_the_window(throttle: AttemptThrottle, clock: FakeClock) -> None:
    throttle.record_failure("alice")
    throttle.record_failure("alice")
    clock.advance(60)

    throttle.record_success("alice")

    assert throttle.retry_after("alice") == 0.0
    assert throttle.record_failure("alice") == 1.0


def test_success_for_an_untracked_user_is_a_no_op(throttle: AttemptThrottle) -> None:
    throttle.record_success("never-failed")

    assert throttle.retry_after("never-failed") == 0.0
    assert len(throttle) == 0


def test_throttling_is_per_user(throttle: AttemptThrottle) -> None:
    """One operator's fat-fingering must not delay anybody else's login."""
    throttle.record_failure("alice")

    assert throttle.retry_after("alice") > 0
    assert throttle.retry_after("bob") == 0.0


def test_retry_after_does_not_extend_the_window(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    """Refused attempts are a pure query — a parallel flood cannot ratchet the window."""
    throttle.record_failure("alice")
    clock.advance(0.5)

    for _ in range(100):
        assert throttle.retry_after("alice") == pytest.approx(0.5)

    clock.advance(0.5)
    assert throttle.retry_after("alice") == 0.0


def test_parallel_attempts_are_all_refused_within_one_window(
    throttle: AttemptThrottle,
) -> None:
    """Throughput-bounded, not latency-delayed: every concurrent guess is refused."""
    throttle.record_failure("alice")

    assert all(throttle.retry_after("alice") > 0 for _ in range(100))


def test_a_quiet_user_is_forgotten_and_starts_over(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    throttle.record_failure("alice")
    throttle.record_failure("alice")

    clock.advance(1000)

    assert throttle.record_failure("alice") == 1.0


def test_stale_users_are_swept_so_memory_stays_bounded(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    for index in range(50):
        throttle.record_failure(f"probe-{index}")
    assert len(throttle) == 50

    clock.advance(1000)
    throttle.record_failure("alice")

    assert len(throttle) == 1


def test_purge_stale_leaves_recently_active_users(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    throttle.record_failure("old")
    clock.advance(1000)
    throttle.record_failure("recent")

    assert throttle.purge_stale() == 0
    assert throttle.retry_after("recent") > 0


def test_purge_stale_keeps_a_user_still_inside_the_forget_horizon(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    throttle.record_failure("alice")
    clock.advance(200)

    assert throttle.purge_stale() == 0
    assert len(throttle) == 1


def test_clear_forgets_every_user(throttle: AttemptThrottle) -> None:
    throttle.record_failure("alice")
    throttle.record_failure("bob")

    throttle.clear()

    assert len(throttle) == 0
    assert throttle.retry_after("alice") == 0.0


def test_sustained_guessing_is_bounded_by_the_cap(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    """An attacker who keeps guessing gets at most one evaluated attempt per cap."""
    evaluated = 0
    for _ in range(600):  # 600 one-second ticks = 10 minutes of hammering
        if throttle.retry_after("alice") == 0.0:
            evaluated += 1
            throttle.record_failure("alice")
        clock.advance(1.0)

    assert evaluated <= 600 / throttle.max_delay + 5


def test_no_lockout_however_long_the_attack_runs(
    throttle: AttemptThrottle, clock: FakeClock
) -> None:
    """The operator is never more than the cap away from their next try."""
    for _ in range(200):
        throttle.record_failure("alice")
        clock.advance(0.1)

    assert throttle.retry_after("alice") <= 30.0

    clock.advance(30.0)
    assert throttle.retry_after("alice") == 0.0

    throttle.record_success("alice")
    assert throttle.retry_after("alice") == 0.0


def test_custom_growth_parameters_are_honoured(clock: FakeClock) -> None:
    throttle = AttemptThrottle(initial_delay=2.0, multiplier=3.0, max_delay=10.0, clock=clock)

    assert throttle.record_failure("alice") == 2.0
    assert throttle.record_failure("alice") == 6.0
    assert throttle.record_failure("alice") == 10.0
    assert throttle.record_failure("alice") == 10.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_delay": 0},
        {"initial_delay": -1.0},
        {"multiplier": 0.5},
        {"max_delay": 0.5},
        {"forget_after": -1.0},
    ],
)
def test_nonsensical_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AttemptThrottle(**kwargs)


def test_defaults_to_a_monotonic_clock() -> None:
    """Durations must not follow wall-clock steps."""
    throttle = AttemptThrottle()

    assert throttle.record_failure("alice") == 1.0
    assert 0.0 < throttle.retry_after("alice") <= 1.0
    throttle.record_success("alice")
    assert throttle.retry_after("alice") == 0.0
