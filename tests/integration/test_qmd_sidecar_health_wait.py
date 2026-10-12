"""The sidecar health wait: renewed while the log advances, failed on silence or exit.

Both sidecar fixtures wait for ``/health`` the same way. A clock and a fake
container stand in for the sidecar, so no image is needed.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from tests.integration import _qmd_ariel_support, _qmd_okf_support

SUPPORTS = (_qmd_ariel_support, _qmd_okf_support)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Wrapped:
    def __init__(self) -> None:
        self.status = "running"

    def reload(self) -> None:
        pass


class _Sidecar:
    """Writes ``per_tick`` bytes of log a second until ``quiet_after``; healthy from ``healthy_at``."""

    def __init__(
        self, clock: _Clock, *, healthy_at: float, quiet_after: float, exits_at: float | None = None
    ) -> None:
        self.clock = clock
        self.healthy_at = healthy_at
        self.quiet_after = quiet_after
        self.exits_at = exits_at
        self.wrapped = _Wrapped()

    def get_logs(self) -> tuple[bytes, bytes]:
        written = int(min(self.clock.now, self.quiet_after))
        return b"#" * written, b""

    def get_wrapped_container(self) -> _Wrapped:
        if self.exits_at is not None and self.clock.now >= self.exits_at:
            self.wrapped.status = "exited"
        return self.wrapped


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    for support in SUPPORTS:
        monkeypatch.setattr(support.time, "monotonic", fake.monotonic)
        monkeypatch.setattr(support.time, "sleep", fake.sleep)
    return fake


def _health(clock: _Clock, sidecar: _Sidecar) -> Any:
    def get(_url: str, timeout: float) -> Any:  # noqa: ARG001 - requests.get's keyword
        if clock.now < sidecar.healthy_at:
            raise requests.ConnectionError("refused")
        return type("Response", (), {"ok": True, "text": '{"status": "ok"}', "status_code": 200})

    return get


@pytest.mark.parametrize("support", SUPPORTS, ids=["ariel", "okf"])
def test_a_slow_start_that_keeps_logging_is_waited_for(
    support: Any, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = support.SIDECAR_HEALTH_TIMEOUT
    sidecar = _Sidecar(clock, healthy_at=2.5 * budget, quiet_after=3 * budget)
    monkeypatch.setattr(support.requests, "get", _health(clock, sidecar))

    support.wait_for_health(sidecar, 1)

    assert clock.now >= 2.5 * budget


@pytest.mark.parametrize("support", SUPPORTS, ids=["ariel", "okf"])
def test_a_silent_sidecar_fails_after_the_budget(
    support: Any, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = support.SIDECAR_HEALTH_TIMEOUT
    sidecar = _Sidecar(clock, healthy_at=10 * budget, quiet_after=10.0)
    monkeypatch.setattr(support.requests, "get", _health(clock, sidecar))

    with pytest.raises(AssertionError, match="wrote nothing"):
        support.wait_for_health(sidecar, 1)

    assert clock.now == pytest.approx(10.0 + budget, abs=2.0)


@pytest.mark.parametrize("support", SUPPORTS, ids=["ariel", "okf"])
def test_an_exited_sidecar_fails_at_once(
    support: Any, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = support.SIDECAR_HEALTH_TIMEOUT
    sidecar = _Sidecar(clock, healthy_at=10 * budget, quiet_after=10 * budget, exits_at=5.0)
    monkeypatch.setattr(support.requests, "get", _health(clock, sidecar))

    with pytest.raises(AssertionError, match="exited before answering"):
        support.wait_for_health(sidecar, 1)

    assert clock.now < budget
