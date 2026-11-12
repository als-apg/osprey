"""Writes queued when a cycle starts are served by that cycle's one pass.

A client that applies many settings at once queues one put per setting, and
each put completes only after the pass that served it. The runner opens no
batching window, so a write that arrives alone is served at once; the writes
already waiting behind it when its cycle starts are merged into that cycle:
one ``model.set``, one readback, one post, and every put completed by it.

Merging never reorders. It stops at the first queued item that is not a plain
write, and at a second write to a channel the pass already holds, and that item
is served next as it was queued. A merged pass that fails is rolled back and its
writes are served one cycle each, so every put gets the verdict it would have
had alone.

The tests drive the runner's own cycle over the base class's, on a hand-made
model with the base class's publishing step recorded, so they need the serving
package but no transport.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

READBACK = "ZZ:RB"
SETTINGS = tuple(f"ZZ:SP{index:03d}" for index in range(166))
FIRST, SECOND, THIRD = SETTINGS[:3]


def _serving() -> Any:
    pytest.importorskip(
        "lume_pva_apg", reason="the served VA runs on lume-pva-apg, which ships for Linux only"
    )
    from lume_pva_apg.runner import Runner

    from osprey.services.virtual_accelerator.serving import runner as module

    return Runner, module


class _Model:
    """Settings and one readback; logs every call, refuses and fails on request."""

    def __init__(self) -> None:
        self.supported_variables = {
            **{name: SimpleNamespace(read_only=False) for name in SETTINGS},
            READBACK: SimpleNamespace(read_only=True),
        }
        self.values: dict[str, Any] = dict.fromkeys(self.supported_variables, 0.0)
        self.log: list[tuple[str, Any]] = []
        self.refused: set[str] = set()
        self.failing_readbacks = 0

    def get(self, names: list[str]) -> dict[str, Any]:
        if READBACK in names:
            self.log.append(("readback", sorted(names)))
            if self.failing_readbacks:
                self.failing_readbacks -= 1
                raise RuntimeError("the readback failed")
        return {name: self.values[name] for name in names}

    def set(self, values: dict[str, Any]) -> None:
        self.log.append(("set", dict(values)))
        for name in values:
            if name in self.refused:
                raise ValueError(f"{name} refused")
        self.values.update(values)

    def restore(self, values: dict[str, Any]) -> None:
        self.log.append(("restore", dict(values)))
        self.values.update(values)

    def reset(self) -> None:
        self.log.append(("reset", None))

    def calls(self, kind: str) -> list[Any]:
        return [detail for logged, detail in self.log if logged == kind]


class _Surface:
    def __init__(self) -> None:
        self.recorded: list[str | None] = []

    def record_pass(self, error: str | None) -> dict[str, Any]:
        self.recorded.append(error)
        return {}


def _runner(module: Any, model: _Model) -> Any:
    """A runner over ``model`` with the state ``__init__`` and the base class would set."""
    runner = object.__new__(module.ModelRunner)
    runner.model = model
    runner.queue = queue.Queue()
    runner.update_rate = 0.0
    runner.pv_handlers = {}
    runner._udf = frozenset()
    runner._tick_interval_s = None
    runner._tick_lock = threading.Lock()
    runner._cycle_requested_names = ()
    runner._periodic = frozenset()
    runner._setting_records = {}
    runner._fast = frozenset()
    runner._timed_names = frozenset()
    runner._needs_timed = False
    runner._forced = frozenset()
    runner._last_ok = True
    runner._served = {}
    runner._written = frozenset()
    runner._write_pass = False
    runner._set_landed = False
    runner._pass_started = False
    runner._passes_run = 0
    runner._first_pass_error = None
    runner._health_file = None
    runner._surface = _Surface()
    return runner


@pytest.fixture
def posts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every base-class post's values."""
    Runner, _ = _serving()
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        Runner, "_post_outputs", lambda _self, out_values, _ts: recorded.append(dict(out_values))
    )
    return recorded


def _put(runner: Any, values: dict[str, Any]) -> list[str | None]:
    """Queue one client put of ``values``; the verdicts its completion was called with."""
    verdicts: list[str | None] = []
    runner._enqueue(
        {name: {"value": value, "ts": 1.0} for name, value in values.items()},
        done=verdicts.append,
    )
    return verdicts


def _serve_queue(runner: Any) -> None:
    """Run the loop until the queue is empty."""
    while not runner.queue.empty():
        runner._run_cycle(runner.queue.get())


def test_writes_queued_before_a_cycle_are_served_by_one_pass(posts: list[Any]) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)
    verdicts = [_put(runner, {name: float(index + 1)}) for index, name in enumerate(SETTINGS)]

    _serve_queue(runner)

    assert len(model.calls("set")) == 1
    assert len(model.calls("readback")) == 1
    assert len(posts) == 1
    assert verdicts == [[None]] * len(SETTINGS)
    assert [model.values[name] for name in SETTINGS] == [
        float(index + 1) for index in range(len(SETTINGS))
    ]
    assert runner._surface.recorded == [None]


def test_a_write_queued_alone_is_served_at_once_and_a_later_one_by_its_own_pass(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)

    first = _put(runner, {FIRST: 1.0})
    runner._run_cycle(runner.queue.get())
    second = _put(runner, {SECOND: 2.0})
    runner._run_cycle(runner.queue.get())

    assert model.calls("set") == [{FIRST: 1.0}, {SECOND: 2.0}]
    assert len(posts) == 2
    assert (first, second) == ([None], [None])


def _queue_reset(runner: Any, _ran: list[str]) -> None:
    runner._enqueue({}, reset=True)


def _queue_tick(runner: Any, _ran: list[str]) -> None:
    runner._tick()


def _queue_job(runner: Any, ran: list[str]) -> None:
    runner._enqueue({}, jobs=[lambda: ran.append("job")])


def _queue_pass(runner: Any, _ran: list[str]) -> None:
    runner._enqueue({}, done=lambda _error: None)


@pytest.mark.parametrize(
    ("queue_other", "between"),
    [
        (_queue_reset, [("reset", None), ("set", {})]),
        (_queue_tick, [("set", {})]),
        (_queue_job, []),
        (_queue_pass, [("set", {})]),
    ],
    ids=["reset", "tick", "job", "pass"],
)
def test_merging_stops_at_an_item_that_is_not_a_plain_write_and_serves_it_next(
    posts: list[Any],
    queue_other: Callable[[Any, list[str]], None],
    between: list[tuple[str, Any]],
) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)
    ran: list[str] = []
    model_set = model.set
    model.set = lambda values: (ran.append("set"), model_set(values))[1]  # type: ignore[method-assign]

    verdicts = [_put(runner, {FIRST: 1.0}), _put(runner, {SECOND: 2.0})]
    queue_other(runner, ran)
    verdicts.append(_put(runner, {THIRD: 3.0}))
    _serve_queue(runner)

    served = [entry for entry in model.log if entry[0] in ("set", "reset")]
    assert served == [("set", {FIRST: 1.0, SECOND: 2.0}), *between, ("set", {THIRD: 3.0})]
    assert verdicts == [[None]] * 3
    assert len(posts) == len(model.calls("set"))
    if queue_other is _queue_job:
        assert ran == ["set", "job", "set"]
    assert runner._tick_pending is False


def test_a_write_that_carries_a_job_is_merged_into_no_pass(posts: list[Any]) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)
    ran: list[str] = []

    _put(runner, {FIRST: 1.0})
    runner._enqueue({SECOND: {"value": 2.0, "ts": 1.0}}, jobs=[lambda: ran.append("job")])
    _put(runner, {THIRD: 3.0})
    _serve_queue(runner)

    assert model.calls("set") == [{FIRST: 1.0}, {SECOND: 2.0}, {THIRD: 3.0}]
    assert ran == ["job"]
    assert len(posts) == 3


def test_a_second_write_to_a_channel_is_a_later_pass(posts: list[Any]) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)

    verdicts = [
        _put(runner, {FIRST: 1.0}),
        _put(runner, {SECOND: 2.0}),
        _put(runner, {FIRST: 3.0}),
        _put(runner, {THIRD: 4.0}),
    ]
    _serve_queue(runner)

    assert model.calls("set") == [{FIRST: 1.0, SECOND: 2.0}, {FIRST: 3.0, THIRD: 4.0}]
    assert len(posts) == 2
    assert verdicts == [[None]] * 4
    assert model.values[FIRST] == 3.0


def test_a_refused_merged_pass_gives_each_put_the_verdict_it_would_have_alone(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model()
    model.refused = {SECOND}
    runner = _runner(module, model)

    first = _put(runner, {FIRST: 1.0})
    second = _put(runner, {SECOND: 2.0})
    third = _put(runner, {THIRD: 3.0})
    _serve_queue(runner)

    assert model.calls("set") == [
        {FIRST: 1.0, SECOND: 2.0, THIRD: 3.0},
        {FIRST: 1.0},
        {SECOND: 2.0},
        {THIRD: 3.0},
    ]
    assert (first, second, third) == ([None], [f"{SECOND} refused"], [None])
    assert (model.values[FIRST], model.values[SECOND], model.values[THIRD]) == (1.0, 0.0, 3.0)
    assert model.calls("restore") == []
    assert len(posts) == 2
    assert runner._surface.recorded == [None, f"{SECOND} refused", None]


def test_a_merged_pass_that_fails_after_its_set_is_restored_then_served_one_cycle_each(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model()
    model.failing_readbacks = 1
    runner = _runner(module, model)

    verdicts = [_put(runner, {FIRST: 1.0}), _put(runner, {SECOND: 2.0})]
    runner._enqueue({}, reset=True)
    _serve_queue(runner)

    assert [entry for entry in model.log if entry[0] != "readback"] == [
        ("set", {FIRST: 1.0, SECOND: 2.0}),
        ("restore", {FIRST: 0.0, SECOND: 0.0}),
        ("set", {FIRST: 1.0}),
        ("set", {SECOND: 2.0}),
        ("reset", None),
        ("set", {}),
    ]
    assert verdicts == [[None], [None]]
    assert len(posts) == 3
    assert runner._surface.recorded == [None, None, None]


def test_a_put_whose_completion_raises_does_not_keep_the_others_waiting(
    posts: list[Any],
) -> None:
    _, module = _serving()
    model = _Model()
    runner = _runner(module, model)

    def broken(_error: str | None) -> None:
        raise RuntimeError("the client went away")

    runner._enqueue({FIRST: {"value": 1.0, "ts": 1.0}}, done=broken)
    second = _put(runner, {SECOND: 2.0})
    _serve_queue(runner)

    assert model.calls("set") == [{FIRST: 1.0, SECOND: 2.0}]
    assert second == [None]
    assert len(posts) == 1
