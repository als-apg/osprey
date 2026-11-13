"""The runner's timed passes, and one publication per emission, without a transport.

:class:`~osprey.services.virtual_accelerator.serving.runner.ModelRunner`
queues a timed pass for a view holding a ``fast`` or ``event`` channel: a
jobs-only item that advances the composite, reads the ``fast`` names and the
emitted ones, and posts them with a forced Channel Access event. Every other
pass drains the emitted names too, so a step is posted once, by whichever pass
advanced over it. These tests drive the runner's own methods on a hand-made
model with the base class's publishing step recorded, so they need the serving
package but no server; the live suite, ``test_timed_passes_live.py``, checks
the same from the far side of both wires.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

FAST = "ZZ:FAST"
EVENT = "ZZ:EVENT"
PASSED = "ZZ:PASS"
SETTING = "ZZ:SP"

VALUES = {FAST: 1.0, EVENT: 2.0, PASSED: 3.0}


def _serving() -> Any:
    pytest.importorskip(
        "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
    )
    from lume_pva_apg.runner import Runner

    from osprey.services.virtual_accelerator.serving import runner as module

    return Runner, module


class _Model:
    """The composite's evolution surface over a fixed set of values."""

    def __init__(self, emitted: tuple[str, ...] = ()) -> None:
        self.values = dict(VALUES)
        self.pending = set(emitted)
        self.advanced = 0
        self.deadline: float | None = None
        self.restored: list[dict[str, Any]] = []
        self.set_values: list[dict[str, Any]] = []
        self.reads: list[list[str]] = []
        self.failure: str | None = None
        self._faulted: dict[str, Any] = {}

    def advance(self) -> None:
        self.advanced += 1

    def emitted(self) -> frozenset[str]:
        drained = frozenset(self.pending)
        self.pending.clear()
        return drained

    def next_deadline(self) -> float | None:
        return self.deadline

    def get(self, names: list[str]) -> dict[str, Any]:
        if self.failure is not None:
            raise RuntimeError(self.failure)
        self.reads.append(list(names))
        return {name: self.values[name] for name in names}

    def set(self, values: dict[str, Any]) -> None:
        self.set_values.append(values)

    def restore(self, values: dict[str, Any]) -> None:
        self.restored.append(values)


class _Surface:
    def __init__(self) -> None:
        self.recorded: list[str | None] = []

    def record_pass(self, error: str | None) -> dict[str, Any]:
        self.recorded.append(error)
        return {}


def _runner(
    module: Any,
    model: _Model,
    *,
    fast: tuple[str, ...] = (FAST,),
    event: tuple[str, ...] = (EVENT,),
    fast_interval_s: float = 0.1,
) -> Any:
    """A runner over ``model`` with the state ``__init__`` and the base class would set."""
    runner = object.__new__(module.ModelRunner)
    runner.model = model
    runner.queue = queue.Queue()
    runner._udf = frozenset()
    runner._tick_interval_s = None
    runner._cycle_requested_names = ()
    runner._fast_interval_s = fast_interval_s
    runner._fast = frozenset(fast)
    runner._timed_names = frozenset(fast) | frozenset(event)
    runner._needs_timed = bool(fast or event)
    runner._forced = frozenset()
    runner._last_ok = True
    runner._deadline = None
    runner._timed_lock = threading.Lock()
    runner._timed_pending = False
    runner._scheduler = None
    runner._scheduler_stop = None
    runner._rearm = threading.Event()
    runner._served = dict(VALUES)
    runner._written = frozenset()
    runner._write_pass = False
    runner._set_landed = False
    runner._health_file = None
    runner._surface = _Surface()
    return runner


@pytest.fixture
def posts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, Any], frozenset[str]]]:
    """Every base-class post, as (values, the names forced for it)."""
    Runner, _ = _serving()
    recorded: list[tuple[dict[str, Any], frozenset[str]]] = []

    def post(self: Any, out_values: dict[str, Any], _ts: float) -> None:
        recorded.append((dict(out_values), frozenset(self._ca_forced_names())))

    monkeypatch.setattr(Runner, "_post_outputs", post)
    return recorded


def test_a_step_due_during_a_tick_pass_is_posted_once(posts: list[Any]) -> None:
    _, module = _serving()
    model = _Model(emitted=(EVENT,))
    runner = _runner(module, model)
    runner._pass_is_tick = True

    module.ModelRunner._post_outputs(runner, dict(VALUES), 0.0)
    runner._timed_pass()

    tick, timed = posts
    assert tick == ({EVENT: 2.0}, frozenset({EVENT}))
    assert timed == ({FAST: 1.0}, frozenset({FAST}))
    assert sum(EVENT in values for values, _ in posts) == 1


def test_a_tick_pass_posts_a_changed_fast_channel_without_forcing_it(posts: list[Any]) -> None:
    _, module = _serving()
    runner = _runner(module, _Model())
    runner._pass_is_tick = True

    module.ModelRunner._post_outputs(runner, {**VALUES, FAST: 5.0}, 0.0)

    assert posts == [({FAST: 5.0}, frozenset())]
    assert runner._served[FAST] == 5.0


def test_a_tick_pass_posts_only_what_changed_since_it_was_last_posted(posts: list[Any]) -> None:
    _, module = _serving()
    runner = _runner(module, _Model(), fast=(), event=())
    runner._pass_is_tick = True

    module.ModelRunner._post_outputs(runner, {**VALUES, PASSED: 4.0}, 0.0)
    module.ModelRunner._post_outputs(runner, {**VALUES, PASSED: 4.0}, 0.0)

    assert posts == [({PASSED: 4.0}, frozenset()), ({}, frozenset())]


def test_a_pass_with_no_inputs_that_is_no_tick_posts_every_value(posts: list[Any]) -> None:
    # Start-up, a reset and a model RPC write publish the whole roster.
    _, module = _serving()
    runner = _runner(module, _Model(), fast=(), event=())

    module.ModelRunner._post_outputs(runner, dict(VALUES), 0.0)

    assert posts == [(dict(VALUES), frozenset())]


def test_an_emitted_name_a_write_pass_did_not_read_is_read_and_posted(posts: list[Any]) -> None:
    _, module = _serving()
    model = _Model(emitted=(EVENT,))
    runner = _runner(module, model)
    runner._write_pass = True
    runner._written = frozenset({SETTING})
    runner._cycle_requested_names = (PASSED,)

    module.ModelRunner._post_outputs(runner, {PASSED: 3.0}, 0.0)

    assert posts == [({EVENT: 2.0}, frozenset({EVENT}))]
    assert model.reads == [[EVENT]]
    assert EVENT in runner._cycle_requested_names


def test_a_timed_pass_advances_reads_fast_and_emitted_names_and_forces_them(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model(emitted=(EVENT,))
    runner = _runner(module, model)
    runner._timed_pending = True
    model.values[EVENT] = 4.0

    runner._timed_pass()

    assert model.advanced == 1
    assert model.reads == [[EVENT, FAST]]
    assert posts == [({EVENT: 4.0, FAST: 1.0}, frozenset({EVENT, FAST}))]
    assert runner._served[EVENT] == 4.0
    assert runner._timed_pending is False
    assert not hasattr(runner, "_cached_state")
    assert runner._forced == frozenset()


def test_a_timed_pass_queued_for_a_deadline_alone_reads_only_the_emitted_names(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model(emitted=(EVENT,))
    runner = _runner(module, model)
    model.values[EVENT] = 4.0

    runner._timed_pass(fast_due=False)

    assert model.reads == [[EVENT]]
    assert posts == [({EVENT: 4.0}, frozenset({EVENT}))]


OTHERS = {f"ZZ:OTHER{index}": float(index) for index in range(3)}


def test_a_post_longer_than_a_slice_goes_out_in_slices_with_the_timed_names_first(
    posts: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = _serving()
    monkeypatch.setattr(module, "POST_SLICE", 2)
    runner = _runner(module, _Model())
    runner._fast_posted_at = time.time()
    runner._cycle_requested_names = ("kept",)

    runner._post_forced({**OTHERS, EVENT: 2.0}, 0.0, frozenset({EVENT}))

    first, second = posts
    assert first == ({EVENT: 2.0, "ZZ:OTHER0": 0.0}, frozenset({EVENT}))
    assert second == ({"ZZ:OTHER1": 1.0, "ZZ:OTHER2": 2.0}, frozenset())
    assert runner._cycle_requested_names == ("kept",)
    assert runner._slicing is False


def test_a_timed_pass_due_between_slices_runs_there_and_its_names_are_not_posted_again(
    posts: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = _serving()
    monkeypatch.setattr(module, "POST_SLICE", 2)
    model = _Model(emitted=(PASSED,))
    model.values[PASSED] = 9.0
    runner = _runner(module, model)

    runner._post_forced(
        {"ZZ:OTHER0": 0.0, "ZZ:OTHER1": 1.0, PASSED: 3.0, "ZZ:OTHER2": 2.0}, 0.0, frozenset()
    )

    assert posts == [
        ({"ZZ:OTHER0": 0.0, "ZZ:OTHER1": 1.0}, frozenset()),
        ({FAST: 1.0, PASSED: 9.0}, frozenset({FAST, PASSED})),
        ({"ZZ:OTHER2": 2.0}, frozenset()),
    ]
    assert model.advanced == 1
    assert runner._served[PASSED] == 9.0


def test_no_timed_pass_runs_between_slices_before_one_is_due(
    posts: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = _serving()
    monkeypatch.setattr(module, "POST_SLICE", 2)
    model = _Model()
    runner = _runner(module, model)
    runner._fast_posted_at = time.time()

    runner._post_forced(dict(OTHERS), 0.0, frozenset())

    assert [set(values) for values, _ in posts] == [{"ZZ:OTHER0", "ZZ:OTHER1"}, {"ZZ:OTHER2"}]
    assert model.advanced == 0


def test_a_view_with_no_timed_channel_posts_whole(
    posts: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = _serving()
    monkeypatch.setattr(module, "POST_SLICE", 2)
    runner = _runner(module, _Model(), fast=(), event=())

    runner._post_forced(dict(OTHERS), 0.0, frozenset())

    assert posts == [(OTHERS, frozenset())]


def test_a_queued_timed_pass_leaves_fast_names_a_sliced_post_just_served(posts: list[Any]) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)
    runner._timed_pending = True
    runner._fast_sliced_at = time.time()

    runner._timed_pass()

    assert model.advanced == 1
    assert model.reads == []
    assert all(FAST not in values for values, _ in posts)
    assert runner._timed_pending is False


def test_a_timed_pass_leaves_a_read_faulted_name_unforced_and_unposted_while_unchanged(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model()
    model._faulted = {FAST: SimpleNamespace(read=lambda value_type, snapshot: snapshot)}
    runner = _runner(module, model)

    runner._timed_pass()
    model.values[FAST] = 7.0
    runner._timed_pass()

    assert posts == [({}, frozenset()), ({FAST: 7.0}, frozenset())]


@pytest.mark.usefixtures("posts")
def test_a_timed_pass_records_its_outcome_only_when_it_changes() -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)

    runner._timed_pass()
    model.failure = "the engine raised"
    runner._timed_pass()
    runner._timed_pass()
    model.failure = None
    runner._timed_pass()
    runner._timed_pass()

    assert runner._surface.recorded == ["the engine raised", None]
    assert runner._timed_pending is False


@pytest.mark.usefixtures("posts")
def test_a_timed_pass_takes_the_next_deadline_and_wakes_the_scheduler() -> None:
    _, module = _serving()
    model = _Model()
    model.deadline = 123.0
    runner = _runner(module, model)

    runner._timed_pass()

    assert runner._deadline == 123.0
    assert runner._rearm.is_set()


def test_a_view_with_neither_class_starts_no_scheduler_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Runner, module = _serving()
    monkeypatch.setattr(Runner, "_start_ticker", lambda self: None)
    monkeypatch.setattr(Runner, "_stop_ticker", lambda self: None)
    runner = _runner(module, _Model(), fast=(), event=(), fast_interval_s=0.01)

    runner._start_ticker()
    time.sleep(0.2)
    names = {thread.name for thread in threading.enumerate()}
    runner._stop_ticker()

    assert runner._scheduler is None
    assert module.SCHEDULER_THREAD not in names
    assert runner.queue.empty()


def test_a_view_with_a_timed_class_queues_one_timed_pass_at_a_time_until_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Runner, module = _serving()
    monkeypatch.setattr(Runner, "_start_ticker", lambda self: None)
    monkeypatch.setattr(Runner, "_stop_ticker", lambda self: None)
    runner = _runner(module, _Model(), fast_interval_s=0.02)

    runner._start_ticker()
    try:
        item = runner.queue.get(timeout=5.0)
        time.sleep(0.2)
        queued_while_pending = runner.queue.qsize()
    finally:
        runner._stop_ticker()

    (job,) = item["jobs"]
    assert item["values"] == {}
    assert (job.func, job.args) == (runner._timed_pass, (True,))
    assert queued_while_pending == 0
    assert runner._scheduler is None
    assert module.SCHEDULER_THREAD not in {thread.name for thread in threading.enumerate()}


def test_a_deadline_sooner_than_the_fast_period_queues_the_timed_pass_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Runner, module = _serving()
    monkeypatch.setattr(Runner, "_start_ticker", lambda self: None)
    monkeypatch.setattr(Runner, "_stop_ticker", lambda self: None)
    runner = _runner(module, _Model(), fast_interval_s=30.0)
    runner._deadline = time.time() + 0.05

    runner._start_ticker()
    try:
        item = runner.queue.get(timeout=5.0)
    finally:
        runner._stop_ticker()

    (job,) = item["jobs"]
    assert (job.func, job.args) == (runner._timed_pass, (False,))


def test_a_failed_write_pass_rolls_back_through_restore() -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)
    runner._cached_state = {SETTING: 1.0}

    runner._reset_to_cached_state()
    runner._set_landed = True
    runner._reset_to_cached_state()

    assert model.restored == [{SETTING: 1.0}]
    assert model.set_values == []


@pytest.mark.parametrize("period", [0, 0.0, -0.1, True, "0.1", None])
def test_a_fast_period_not_greater_than_zero_is_refused(period: Any) -> None:
    _, module = _serving()

    with pytest.raises(ValueError, match="the fast period must be a number greater than zero"):
        module.ModelRunner(None, None, model_write_token=None, fast_interval_s=period)
