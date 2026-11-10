"""The orbit-bump ORM cache: one background-built response matrix per model.

A fake SDK stands in for ``osprey.apps.sdk``: each build blocks on a gate the test
opens, so the test sees ``building`` and checks that state reads never wait for it.
"""

import threading

import numpy as np
import pytest
from orbit_bump.orm import BUILDING, FAILED, IDLE, READY, OrmCache, OrmSnapshot

from osprey.apps.sdk import ResponseMatrix

WAIT = 5.0
ROWS = ["B1:X", "B2:X", "B1:Y"]
COLUMNS = ["C1:X", "C2:X", "C1:Y"]
MATRIX = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


class FakeSdk:
    """The three SDK names the cache calls, with builds the test releases by hand."""

    def __init__(self, models=("RING", "OTHER")):
        self.models = list(models)
        self.calls: list[tuple[str, object, float]] = []
        self.outcomes: dict[str, list[object]] = {}
        self.gates: dict[str, threading.Event] = {}
        self.entered = threading.Semaphore(0)

    def closed_orbit_models(self):
        return list(self.models)

    def queue(self, model, *outcomes):
        self.outcomes.setdefault(model, []).extend(outcomes)

    def gate(self, model):
        return self.gates.setdefault(model, threading.Event())

    def model_response_matrix(self, model, correctors=None, render_root=None, timeout_s=120):  # noqa: ARG002
        self.calls.append((model, correctors, timeout_s))
        self.entered.release()
        assert self.gate(model).wait(WAIT), "the test never released the build"
        outcome = self.outcomes[model].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _matrix(model, scale=1.0, unsolved=None):
    return ResponseMatrix(
        model=model,
        rows=list(ROWS),
        columns=list(COLUMNS),
        matrix=[[scale * value for value in row] for row in MATRIX],
        unsolved=dict(unsolved or {}),
    )


@pytest.fixture
def sdk():
    return FakeSdk()


@pytest.fixture
def cache(sdk):
    built = []
    cache = OrmCache(sdk=sdk, timeout_s=42.0, clock=lambda: 1000.0, on_ready=built.append)
    cache.built = built
    return cache


def _entered(sdk):
    assert sdk.entered.acquire(timeout=WAIT), "no build started"


def _finish(cache, sdk, model):
    sdk.gate(model).set()
    assert cache.join(model, WAIT), "the build did not end"


def test_unknown_model_is_idle_with_no_snapshot(cache):
    state = cache.state("RING")
    assert (state.status, state.version, state.built_at, state.error) == (IDLE, 0, None, None)
    assert state.unsolved == {}
    assert cache.snapshot("RING") is None


def test_start_builds_the_first_closed_orbit_model_only(cache, sdk):
    assert cache.start() == "RING"
    _entered(sdk)
    assert cache.state("RING").status == BUILDING
    assert cache.state("OTHER").status == IDLE
    sdk.queue("RING", _matrix("RING"))
    _finish(cache, sdk, "RING")
    assert [call[0] for call in sdk.calls] == ["RING"]
    assert cache.state("OTHER").status == IDLE


def test_start_with_no_closed_orbit_model_builds_nothing():
    sdk = FakeSdk(models=())
    cache = OrmCache(sdk=sdk)
    assert cache.start() is None
    assert sdk.calls == []


def test_success_asks_every_corrector_with_the_timeout_and_swaps_a_snapshot(cache, sdk):
    sdk.queue("RING", _matrix("RING", unsolved={"C9:X": "no element"}))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")

    assert sdk.calls == [("RING", None, 42.0)]
    state = cache.state("RING")
    assert (state.status, state.version, state.built_at, state.error) == (READY, 1, 1000.0, None)
    assert state.unsolved == {"C9:X": "no element"}
    snap = cache.snapshot("RING")
    assert isinstance(snap, OrmSnapshot)
    assert snap.version == 1
    assert snap.rows == tuple(ROWS)
    assert snap.columns == tuple(COLUMNS)
    np.testing.assert_array_equal(snap.matrix, np.array(MATRIX))
    assert snap.unsolved == {"C9:X": "no element"}
    assert cache.built == ["RING"]


def test_snapshot_is_immutable(cache, sdk):
    sdk.queue("RING", _matrix("RING"))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    snap = cache.snapshot("RING")
    with pytest.raises(ValueError):
        snap.matrix[0, 0] = 99.0
    with pytest.raises(AttributeError):
        snap.version = 7  # type: ignore[misc]
    with pytest.raises(TypeError):
        snap.unsolved["X"] = "y"  # type: ignore[index]


def test_slice_returns_the_submatrix_in_the_callers_order(cache, sdk):
    sdk.queue("RING", _matrix("RING"))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    snap = cache.snapshot("RING")
    sub = snap.slice(["B1:Y", "B1:X"], ["C2:X", "C1:X"])
    np.testing.assert_array_equal(sub, np.array([[8.0, 7.0], [2.0, 1.0]]))
    assert sub.shape == (2, 2)
    empty = snap.slice([], ["C1:X"])
    assert empty.shape == (0, 1)


def test_slice_names_an_address_the_matrix_lacks(cache, sdk):
    sdk.queue("RING", _matrix("RING", unsolved={"C9:X": "no element"}))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    snap = cache.snapshot("RING")
    with pytest.raises(KeyError, match="C9:X"):
        snap.slice(["B1:X"], ["C9:X"])
    with pytest.raises(KeyError, match="B7:X"):
        snap.slice(["B7:X"], ["C1:X"])


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (ValueError("the response of RING was refused: no deck"), "refused: no deck"),
        (TimeoutError("the response of RING was not solved within 42 s"), "within 42 s"),
        (RuntimeError("the solve of RING ended without a result"), "without a result"),
    ],
)
def test_refusal_or_timeout_fails_with_the_reason_and_drops_the_old_matrix(cache, sdk, error, text):
    sdk.queue("RING", _matrix("RING"), error)
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    assert cache.snapshot("RING") is not None

    sdk.gate("RING").clear()
    assert cache.recompute("RING") is True
    _finish(cache, sdk, "RING")

    state = cache.state("RING")
    assert state.status == FAILED
    assert text in state.error
    assert state.built_at is None
    assert state.unsolved == {}
    assert state.version == 1
    assert cache.snapshot("RING") is None
    assert cache.built == ["RING"]


def test_failed_entry_is_not_rebuilt_by_ensure_but_is_by_recompute(cache, sdk):
    sdk.queue("RING", ValueError("refused"), _matrix("RING"))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    assert cache.state("RING").status == FAILED

    cache.ensure("RING")
    assert len(sdk.calls) == 1

    assert cache.recompute("RING") is True
    _finish(cache, sdk, "RING")
    state = cache.state("RING")
    assert (state.status, state.version, state.error) == (READY, 1, None)


def test_each_success_bumps_the_version(cache, sdk):
    sdk.queue("RING", _matrix("RING"), _matrix("RING", scale=2.0))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    first = cache.snapshot("RING")

    assert cache.recompute("RING") is True
    _finish(cache, sdk, "RING")
    second = cache.snapshot("RING")

    assert (first.version, second.version) == (1, 2)
    assert cache.state("RING").version == 2
    np.testing.assert_array_equal(first.matrix, np.array(MATRIX))
    np.testing.assert_array_equal(second.matrix, 2.0 * np.array(MATRIX))
    assert cache.built == ["RING", "RING"]


def test_reads_never_block_on_a_build_and_the_old_snapshot_stays_until_the_swap(cache, sdk):
    sdk.queue("RING", _matrix("RING"), _matrix("RING", scale=3.0))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    sdk.gate("RING").clear()

    cache.recompute("RING")
    _entered(sdk)
    _entered(sdk)  # the first build's entry, then the recompute's
    # The build is parked inside the SDK call: every read answers at once.
    assert cache.state("RING").status == BUILDING
    assert cache.state("RING").version == 1
    assert cache.snapshot("RING").version == 1
    assert cache.join("RING", 0.05) is False

    _finish(cache, sdk, "RING")
    assert cache.snapshot("RING").version == 2


def test_ensure_and_recompute_while_building_start_no_second_build(cache, sdk):
    sdk.queue("RING", _matrix("RING"))
    cache.ensure("RING")
    _entered(sdk)
    cache.ensure("RING")
    assert cache.recompute("RING") is False
    _finish(cache, sdk, "RING")
    assert len(sdk.calls) == 1


def test_ready_entry_is_not_rebuilt_by_ensure(cache, sdk):
    sdk.queue("RING", _matrix("RING"))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    assert cache.ensure("RING").status == READY
    assert len(sdk.calls) == 1


def test_second_model_builds_lazily_on_its_first_ensure(cache, sdk):
    sdk.queue("RING", _matrix("RING"))
    sdk.queue("OTHER", _matrix("OTHER", scale=5.0))
    cache.start()
    _finish(cache, sdk, "RING")
    assert cache.state("OTHER").status == IDLE
    assert cache.snapshot("OTHER") is None

    assert cache.ensure("OTHER").status == BUILDING
    _finish(cache, sdk, "OTHER")

    assert [call[0] for call in sdk.calls] == ["RING", "OTHER"]
    assert cache.state("OTHER").version == 1
    np.testing.assert_array_equal(cache.snapshot("OTHER").matrix, 5.0 * np.array(MATRIX))
    assert cache.snapshot("RING").version == 1
    assert cache.built == ["RING", "OTHER"]


def test_recompute_starts_an_idle_models_first_build(cache, sdk):
    sdk.queue("OTHER", _matrix("OTHER"))
    assert cache.recompute("OTHER") is True
    _finish(cache, sdk, "OTHER")
    assert cache.state("OTHER").status == READY


def test_a_raising_on_ready_hook_leaves_the_entry_ready(sdk):
    def hook(_model):
        raise RuntimeError("plan store fell over")

    cache = OrmCache(sdk=sdk, on_ready=hook)
    sdk.queue("RING", _matrix("RING"))
    cache.ensure("RING")
    _finish(cache, sdk, "RING")
    assert cache.state("RING").status == READY


def test_join_on_a_model_never_built_returns_at_once(cache):
    assert cache.join("RING", 0.0) is True
