"""The runner posts one drain of emitted names in one piece, arrays as scalars.

A pass in a view with a timed channel posts in slices, and between two slices
runs a timed pass that fell due. The names one drain of
:meth:`emitted` returned are one step of the composite, so they all go in the
first slice, however many there are: no timed pass can post the next step
between two names of this one. An array an engine emits is posted and forced as
a Channel Access event exactly as a scalar is.

The tests drive the runner's own methods on a hand-made model, with the base
class's publishing step recorded. The serving package ships for Linux only; on
a host without it a stand-in base class carries the members these tests
reach, and the runner module is imported over it for the test alone.
"""

from __future__ import annotations

import importlib
import queue
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

RUNNER_MODULE = "osprey.services.virtual_accelerator.serving.runner"

FAST = "ZZ:FAST"
FLAG = "ZZ:SHOT:DONE"
OTHERS = {f"ZZ:OTHER{index}": float(index) for index in range(4)}
ARRAYS = {f"ZZ:TBT{index}": np.arange(5, dtype=float) + index for index in range(3)}


class _StandInRunner:
    """The members of ``lume_pva_apg.runner.Runner`` these tests reach."""

    # The base class's default: a pass is a periodic one only when it says so.
    _pass_is_tick: bool = False

    def _post_outputs(self, out_values: dict[str, Any], ts: float) -> None:  # noqa: ARG002  # pragma: no cover
        raise AssertionError("the post is recorded by the test")

    def _ca_forced_names(self) -> frozenset[str]:  # pragma: no cover
        return frozenset()


@pytest.fixture
def serving(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, Any]]:
    """The base class and the runner module, over a stand-in where the package is absent."""
    try:
        from lume_pva_apg.runner import Runner
    except ImportError:
        pass
    else:
        yield Runner, importlib.import_module(RUNNER_MODULE)
        return
    package = types.ModuleType("lume_pva_apg")
    runner_module = types.ModuleType("lume_pva_apg.runner")
    runner_module.Runner = _StandInRunner  # type: ignore[attr-defined]
    package.runner = runner_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lume_pva_apg", package)
    monkeypatch.setitem(sys.modules, "lume_pva_apg.runner", runner_module)
    previous = sys.modules.pop(RUNNER_MODULE, None)
    try:
        yield _StandInRunner, importlib.import_module(RUNNER_MODULE)
    finally:
        sys.modules.pop(RUNNER_MODULE, None)
        if previous is not None:
            sys.modules[RUNNER_MODULE] = previous


@pytest.fixture
def posts(
    serving: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> list[tuple[dict[str, Any], frozenset[str]]]:
    """Every base-class post, as (values, the names forced for it)."""
    base, _ = serving
    recorded: list[tuple[dict[str, Any], frozenset[str]]] = []

    def post(self: Any, out_values: dict[str, Any], _ts: float) -> None:
        recorded.append((dict(out_values), frozenset(self._ca_forced_names())))
        hook = getattr(self.model, "after_post", None)
        if hook is not None:
            hook(len(recorded))

    monkeypatch.setattr(base, "_post_outputs", post)
    return recorded


class _Model:
    """A composite's evolution surface over fixed values, with a scripted emission."""

    def __init__(self, values: dict[str, Any], emitted: tuple[str, ...] = ()) -> None:
        self.values = dict(values)
        self.pending = set(emitted)
        self.emit_after_first_post: tuple[str, ...] = ()
        self.advanced = 0
        self.reads: list[list[str]] = []
        self._faulted: dict[str, Any] = {}

    def after_post(self, count: int) -> None:
        if count == 1:
            self.pending |= set(self.emit_after_first_post)

    def advance(self) -> None:
        self.advanced += 1

    def emitted(self) -> frozenset[str]:
        drained = frozenset(self.pending)
        self.pending.clear()
        return drained

    def next_deadline(self) -> float | None:
        return None

    def get(self, names: list[str]) -> dict[str, Any]:
        self.reads.append(list(names))
        return {name: self.values[name] for name in names}


class _Surface:
    def record_pass(self, error: str | None) -> dict[str, Any]:  # noqa: ARG002
        return {}


def _runner(module: Any, model: _Model, *, fast_due: bool) -> Any:
    """A runner over ``model`` with the state ``__init__`` and the base class would set."""
    runner = object.__new__(module.ModelRunner)
    runner.model = model
    runner.queue = queue.Queue()
    runner._udf = frozenset()
    runner._tick_interval_s = None
    runner._cycle_requested_names = ()
    runner._fast_interval_s = 0.1
    runner._fast = frozenset({FAST})
    runner._timed_names = frozenset({FAST, FLAG})
    runner._needs_timed = True
    runner._forced = frozenset()
    runner._last_ok = True
    runner._deadline = None
    runner._timed_lock = threading.Lock()
    runner._timed_pending = False
    runner._rearm = threading.Event()
    runner._served = {}
    runner._written = frozenset()
    runner._write_pass = False
    runner._health_file = None
    runner._surface = _Surface()
    runner._evaluate_severity = lambda names: frozenset()
    # A fast period fell due long ago, or was just served.
    runner._fast_posted_at = 0.0 if fast_due else float("inf")
    return runner


def _index_of(posts: list[Any], name: str) -> int:
    return next(index for index, (values, _) in enumerate(posts) if name in values)


def test_more_emitted_names_than_a_slice_holds_all_go_before_a_due_timed_pass(
    posts: list[Any], serving: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = serving
    monkeypatch.setattr(module, "POST_SLICE", 2)
    emitted = tuple(ARRAYS)
    model = _Model({**OTHERS, **ARRAYS, FAST: 1.0}, emitted=emitted)
    runner = _runner(module, model, fast_due=True)

    module.ModelRunner._post_outputs(runner, {**OTHERS, **ARRAYS}, 0.0)

    first_values, first_forced = posts[0]
    assert set(emitted) <= first_values.keys()
    assert set(emitted) <= first_forced
    timed = _index_of(posts, FAST)
    assert timed > 0
    assert model.advanced == 1
    assert all(_index_of(posts, name) < timed for name in emitted)


def test_an_array_drain_is_posted_before_a_flag_emitted_in_the_next_drain(
    posts: list[Any], serving: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = serving
    monkeypatch.setattr(module, "POST_SLICE", 2)
    model = _Model({**OTHERS, **ARRAYS, FAST: 1.0, FLAG: 1}, emitted=tuple(ARRAYS))
    model.emit_after_first_post = (FLAG,)
    runner = _runner(module, model, fast_due=True)

    module.ModelRunner._post_outputs(runner, {**OTHERS, **ARRAYS}, 0.0)

    flag = _index_of(posts, FLAG)
    assert all(_index_of(posts, name) < flag for name in ARRAYS)
    assert FLAG in posts[flag][1]
    assert sum(FLAG in values for values, _ in posts) == 1


def test_an_array_drain_in_a_timed_pass_goes_in_one_slice(
    posts: list[Any], serving: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = serving
    monkeypatch.setattr(module, "POST_SLICE", 2)
    model = _Model({**ARRAYS, FAST: 1.0, FLAG: 1}, emitted=tuple(ARRAYS))
    runner = _runner(module, model, fast_due=True)

    runner._timed_pass()

    values, forced = posts[0]
    assert set(ARRAYS) <= values.keys()
    assert set(ARRAYS) <= forced


def test_an_emitted_array_reaches_the_post_with_a_forced_event(
    posts: list[Any], serving: tuple[Any, Any]
) -> None:
    _, module = serving
    name, array = next(iter(ARRAYS.items()))
    model = _Model({name: array, FAST: 1.0}, emitted=(name,))
    runner = _runner(module, model, fast_due=False)
    runner._served = {name: array.copy()}

    module.ModelRunner._post_outputs(runner, {name: array}, 0.0)

    assert len(posts) == 1
    values, forced = posts[0]
    assert np.array_equal(values[name], array)
    assert name in forced
    assert np.array_equal(runner._served[name], array)


@pytest.mark.parametrize("kind", ["tick", "write", "timed"])
def test_a_shot_at_the_values_last_posted_is_posted_and_forced_by_any_pass(
    posts: list[Any], serving: tuple[Any, Any], kind: str
) -> None:
    # A shot that reproduces the last one still reaches every client: its
    # arrays and its flag are emitted, and no pass narrows an emitted name away.
    _, module = serving
    shot = {**ARRAYS, FLAG: 0}
    held = {**shot, **OTHERS, FAST: 1.0}
    model = _Model(held, emitted=tuple(shot))
    runner = _runner(module, model, fast_due=False)
    runner._served = {**held, **{name: array.copy() for name, array in ARRAYS.items()}}
    runner._pass_is_tick = kind == "tick"
    runner._write_pass = kind == "write"
    runner._written = frozenset({"ZZ:OTHER0"}) if kind == "write" else frozenset()

    if kind == "timed":
        runner._run_timed(False)
    else:
        module.ModelRunner._post_outputs(runner, dict(held), 0.0)

    posted = {name for values, _ in posts for name in values}
    forced = {name for _, names in posts for name in names}
    assert set(shot) <= posted
    assert set(shot) <= forced
    assert posted - set(shot) == ({"ZZ:OTHER0"} if kind == "write" else set())


def test_an_emitted_array_a_tick_pass_did_not_read_is_read_and_forced(
    posts: list[Any], serving: tuple[Any, Any]
) -> None:
    _, module = serving
    name, array = next(iter(ARRAYS.items()))
    model = _Model({name: array, FAST: 1.0, "ZZ:PASS": 3.0}, emitted=(name,))
    runner = _runner(module, model, fast_due=False)

    module.ModelRunner._post_outputs(runner, {"ZZ:PASS": 3.0}, 0.0)

    values, forced = posts[0]
    assert model.reads == [[name]]
    assert np.array_equal(values[name], array)
    assert forced == frozenset({name})


def test_a_sliced_post_without_emitted_names_slices_as_before(
    posts: list[Any], serving: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, module = serving
    monkeypatch.setattr(module, "POST_SLICE", 2)
    runner = _runner(module, _Model(dict(OTHERS)), fast_due=False)

    runner._post_forced(dict(OTHERS), 0.0, frozenset())

    assert [list(values) for values, _ in posts] == [
        ["ZZ:OTHER0", "ZZ:OTHER1"],
        ["ZZ:OTHER2", "ZZ:OTHER3"],
    ]
