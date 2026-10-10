"""The deadline guard stops a pyAML tool while a restore still fits before the sandbox kill.

With ``OSPREY_EXECUTION_DEADLINE`` set and a method that accepts ``callback``,
``run_tool`` chains a check behind the caller's callback. It returns a strict
``False`` when the caller says stop (pyAML's truthiness rule) or when the time left
is below ``2 * (interval + latency * (walk_back_writes + 1) + read_latency)`` plus
``EXIT_RESERVE_S``. Every clock here is fake: writes and sleeps advance it by hand.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pyaml.common.constants import Action

import osprey.runtime
import osprey.runtime.journal as journal_mod
import pyaml_cs_osprey.run_tool as run_tool_mod
from osprey.runtime.journal import active_journals
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig
from pyaml_cs_osprey.run_tool import (
    DEFAULT_WRITE_LATENCY_S,
    EXIT_RESERVE_S,
    REPORT_TAG,
    RestoreReport,
    run_tool,
)
from tests.pyaml_cs_osprey.conftest import FakeRuntime as _Machine

START = 1_000_000.0


class _Clock:
    """A settable wall clock shared by the guard and the journal."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Tool:
    """A pyAML-shaped measurement: step, sleep, callback; a falsy callback aborts."""

    def __init__(self, machine: _Machine, steps: list[tuple[str, float]]) -> None:
        self.machine = machine
        self.steps = steps
        self.sleep_between_step = 0.0
        self.sleep_between_meas = 0.0
        self.callback_first = False
        self.returned: list[Any] = []

    def _send(self, callback: Callable[..., Any] | None) -> None:
        if callback is None:
            return
        ok = callback(Action.APPLY, {"source": self})
        self.returned.append(ok)
        if not ok:
            raise KeyboardInterrupt

    def measure(
        self,
        sleep_between_step: float | None = None,
        sleep_between_meas: float | None = None,
        callback: Callable[..., Any] | None = None,
    ) -> bool:
        del sleep_between_meas  # the fake takes no measurement between steps
        step_s = self.sleep_between_step if sleep_between_step is None else sleep_between_step
        if self.callback_first:
            self._send(callback)
        for address, value in self.steps:
            self.machine.set(address, value)
            self.machine.clock.advance(step_s)
            self._send(callback)
        return True

    def correct(self) -> None:
        for address, value in self.steps:
            self.machine.set(address, value)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(run_tool_mod, "_clock", c)
    monkeypatch.setattr(journal_mod, "_clock", c)
    return c


@pytest.fixture
def machine(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[_Machine]:
    m = _Machine({"A": 1.0, "B": 2.0, "C": 3.0}, clock=clock)
    yield m.install(monkeypatch, ("read_channels", "write_channel", "channel_limits"))
    assert active_journals() == ()


def _deadline_in(monkeypatch: pytest.MonkeyPatch, seconds: float) -> float:
    deadline = START + seconds
    monkeypatch.setenv(osprey.runtime.ENV_EXECUTION_DEADLINE, repr(deadline))
    return deadline


class TestAbortsInTime:
    def test_a_long_run_aborts_before_the_deadline_and_the_restore_completes(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine, clock: _Clock
    ) -> None:
        machine.write_s = 0.2
        steps = [(addr, 10.0 + k) for k in range(50) for addr in ("A", "B", "C")]
        tool = _Tool(machine, steps)
        deadline = _deadline_in(monkeypatch, 30.0)

        report = run_tool(tool.measure, sleep_between_step=1.0)

        assert report.aborted is True
        assert report.deadline_guard is True
        assert sorted(report.restored) == ["A", "B", "C"]
        assert machine.values == {"A": 1.0, "B": 2.0, "C": 3.0}
        assert tool.returned[-1] is False
        assert clock.now <= deadline - EXIT_RESERVE_S

    def test_the_report_line_of_a_deadline_stop_says_guarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        machine: _Machine,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The one tagged line the run prints is the report returned, guard included."""
        machine.write_s = 0.2
        tool = _Tool(machine, [(addr, 10.0 + k) for k in range(50) for addr in ("A", "B")])
        _deadline_in(monkeypatch, 30.0)

        report = run_tool(tool.measure, sleep_between_step=1.0)

        out, err = capsys.readouterr()
        lines = [
            json.loads(line[len(REPORT_TAG) + 1 :])
            for line in (out + err).splitlines()
            if line.startswith(REPORT_TAG + " ")
        ]
        assert report.aborted is True
        assert report.deadline_guard is True
        assert lines == [json.loads(report.to_json())]

    def test_the_first_step_budgets_the_default_latency_before_any_sample(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        # Callback before any write: empty journal, no latency sample, no gaps.
        threshold = 2.0 * DEFAULT_WRITE_LATENCY_S + EXIT_RESERVE_S
        tool = _Tool(machine, [("A", 5.0)])
        tool.callback_first = True
        _deadline_in(monkeypatch, threshold - 0.1)

        report = run_tool(tool.measure)

        assert tool.returned == [False]
        assert report.aborted is True
        assert report.deadline_guard is True
        assert machine.writes == []

    def test_the_first_step_continues_when_the_default_budget_fits(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        threshold = 2.0 * DEFAULT_WRITE_LATENCY_S + EXIT_RESERVE_S
        tool = _Tool(machine, [])
        tool.callback_first = True
        _deadline_in(monkeypatch, threshold + 0.1)

        report = run_tool(tool.measure)

        assert tool.returned == [True]
        assert report.aborted is False
        assert report.deadline_guard is True

    def test_a_sleep_keyword_overrides_a_zero_attribute_and_still_aborts_in_time(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine, clock: _Clock
    ) -> None:
        machine.write_s = 0.1
        tool = _Tool(machine, [("A", 5.0), ("A", 6.0), ("A", 7.0)])
        assert tool.sleep_between_step == 0.0
        # First callback at +5.1 leaves 13.9 s; the 5 s seed needs about 15.4.
        deadline = _deadline_in(monkeypatch, 19.0)

        report = run_tool(tool.measure, sleep_between_step=5.0)

        assert tool.returned == [False]
        assert report.aborted is True
        assert report.restored == ["A"]
        assert machine.values["A"] == 1.0
        assert clock.now <= deadline - EXIT_RESERVE_S

    def test_the_owner_attributes_seed_the_interval_without_a_keyword(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        machine.write_s = 0.1
        tool = _Tool(machine, [("A", 5.0)])
        tool.sleep_between_meas = 5.0
        # No sleep happens, so only the 5 s attribute seed makes 10 s too little.
        _deadline_in(monkeypatch, 10.0)

        report = run_tool(tool.measure)

        assert tool.returned == [False]
        assert report.aborted is True

    def test_a_three_step_walk_back_is_budgeted(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine, clock: _Clock
    ) -> None:
        machine.write_s = 0.5
        machine.limits["A"] = ChannelLimitsConfig(
            channel_address="A", min_value=-100.0, max_value=100.0, max_step=1.0, writable=True
        )
        # One write back would need 2 * 0.5 * 2 + 5 = 7 s; three need 2 * 0.5 * 4 + 5 = 9 s.
        tool = _Tool(machine, [("A", 4.0)])
        deadline = _deadline_in(monkeypatch, 8.5)

        report = run_tool(tool.measure)

        assert tool.returned == [False]
        assert report.aborted is True
        assert report.restored == ["A"]
        assert machine.writes == [("A", v, {}) for v in (4.0, 3.0, 2.0, 1.0)]
        assert clock.now <= deadline - EXIT_RESERVE_S

    def test_a_run_with_time_to_spare_completes_guarded(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        tool = _Tool(machine, [("A", 5.0), ("B", 6.0)])
        _deadline_in(monkeypatch, 3600.0)

        report = run_tool(tool.measure)

        assert tool.returned == [True, True]
        assert report == RestoreReport(aborted=False, deadline_guard=True)
        assert machine.values["A"] == 5.0


class TestUnguarded:
    def test_a_method_without_callback_reports_no_guard(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        tool = _Tool(machine, [("A", 5.0)])
        _deadline_in(monkeypatch, 0.0)

        report = run_tool(tool.correct)

        assert report.aborted is False
        assert report.deadline_guard is False
        assert machine.values["A"] == 5.0

    def test_a_run_without_the_deadline_variable_reports_no_guard(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        monkeypatch.delenv(osprey.runtime.ENV_EXECUTION_DEADLINE, raising=False)
        tool = _Tool(machine, [("A", 5.0)])

        report = run_tool(tool.measure, callback=lambda _a, _d: True)

        assert report.deadline_guard is False
        assert tool.returned == [True]

    def test_an_aborted_unguarded_run_reports_no_guard(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        monkeypatch.delenv(osprey.runtime.ENV_EXECUTION_DEADLINE, raising=False)
        tool = _Tool(machine, [("A", 5.0)])

        report = run_tool(tool.measure, callback=lambda _a, _d: False)

        assert report.aborted is True
        assert report.deadline_guard is False


class TestCallerCallback:
    @pytest.mark.parametrize("says", [False, None, 0])
    def test_a_falsy_caller_result_aborts_with_a_strict_false(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine, says: Any
    ) -> None:
        tool = _Tool(machine, [("A", 5.0), ("B", 6.0)])
        _deadline_in(monkeypatch, 3600.0)
        seen: list[Any] = []

        def caller(action: Any, data: Any) -> Any:
            seen.append((action, data["source"]))
            return says

        report = run_tool(tool.measure, callback=caller)

        assert seen == [(Action.APPLY, tool)]
        assert tool.returned == [False]
        assert report.aborted is True
        assert report.deadline_guard is True
        assert report.restored == ["A"]
        assert machine.values["A"] == 1.0

    def test_a_truthy_caller_result_continues_with_a_strict_true(
        self, monkeypatch: pytest.MonkeyPatch, machine: _Machine
    ) -> None:
        tool = _Tool(machine, [("A", 5.0)])
        _deadline_in(monkeypatch, 3600.0)

        report = run_tool(tool.measure, callback=lambda _a, _d: "yes")

        assert tool.returned == [True]
        assert report.aborted is False

    @pytest.mark.usefixtures("machine")
    def test_the_chain_is_inert_after_run_tool_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[Callable[..., Any]] = []
        calls: list[Any] = []

        def measure(callback: Callable[..., Any] | None = None) -> bool:
            assert callback is not None
            captured.append(callback)
            return True

        _deadline_in(monkeypatch, 0.0)
        run_tool(measure, callback=lambda a, d: calls.append(a))

        assert captured[0](Action.APPLY, {}) is True
        assert calls == []
