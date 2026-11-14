"""What a write pass of the served composite posts, decided in process.

:func:`~osprey.services.virtual_accelerator.serving.publication.same_value`
says whether two values would reach a client as the same bytes, and
:func:`~osprey.services.virtual_accelerator.serving.publication.changed_outputs`
narrows a pass's values to those a client has not received yet. The identity
case runs the built demo composite and proves that posting only
the changes leaves every client holding what a full post would have given it.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from osprey.services.virtual_accelerator.serving import publication
from osprey.services.virtual_accelerator.serving.publication import (
    changed_outputs,
    forced_names,
    next_wake,
    same_value,
    timed_pass_names,
    untimed_outputs,
)
from osprey.services.virtual_accelerator.serving.runner_config import periodic_addresses
from tests._builds import BuiltProject

# -- same_value ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        pytest.param(0.0, -0.0, False, id="signed-zero"),
        pytest.param(-0.0, -0.0, True, id="negative-zero"),
        pytest.param(math.nan, math.nan, True, id="nan"),
        pytest.param(math.nan, 0.0, False, id="nan-and-number"),
        pytest.param(1.5, 1.5, True, id="equal-floats"),
        pytest.param(1.5, 1.25, False, id="different-floats"),
        pytest.param(True, 1, False, id="bool-is-not-int"),
        pytest.param(1, 1.0, False, id="int-is-not-float"),
        pytest.param(7, 7, True, id="equal-ints"),
        pytest.param("ok", "ok", True, id="equal-strings"),
        pytest.param("ok", "fault", False, id="different-strings"),
    ],
)
def test_same_value_holds_only_for_values_that_pack_alike(a: Any, b: Any, same: bool) -> None:
    assert same_value(a, b) is same


def test_equal_arrays_of_different_dtype_are_not_the_same_value() -> None:
    assert same_value(np.array([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32)) is False
    assert same_value(np.array([1, 2]), np.array([1.0, 2.0])) is False


def test_arrays_of_different_shape_are_not_the_same_value() -> None:
    assert same_value(np.zeros(4), np.zeros((2, 2))) is False


def test_arrays_with_nan_at_the_same_index_are_the_same_value() -> None:
    a = np.array([0.1, math.nan, 0.3])
    assert same_value(a, a.copy()) is True
    assert same_value(a, np.array([0.1, 0.2, 0.3])) is False


def test_a_trace_differing_in_one_sample_is_not_the_same_value() -> None:
    trace = np.linspace(-1.0, 1.0, 500, dtype=np.float64)
    moved = trace.copy()
    moved[317] += 1e-12

    assert same_value(trace, trace.copy()) is True
    assert same_value(trace, moved) is False


# -- changed_outputs -----------------------------------------------------------


def test_a_name_never_served_is_posted() -> None:
    posted = changed_outputs({"A": 1.0}, {}, frozenset(), frozenset(), frozenset())
    assert posted == {"A": 1.0}


def test_a_written_name_with_an_unchanged_value_is_posted() -> None:
    posted = changed_outputs({"SP": 2.0}, {"SP": 2.0}, frozenset({"SP"}), frozenset(), frozenset())
    assert posted == {"SP": 2.0}


def test_an_undefined_condition_that_flips_is_posted_both_ways() -> None:
    served = {"RB": 3.0}

    became_udf = changed_outputs({"RB": 3.0}, served, frozenset(), frozenset({"RB"}), frozenset())
    recovered = changed_outputs({"RB": 3.0}, served, frozenset(), frozenset(), frozenset({"RB"}))

    assert became_udf == {"RB": 3.0}
    assert recovered == {"RB": 3.0}


def test_an_unchanged_name_is_left_out_and_order_is_kept() -> None:
    out = {"C": 3.0, "A": 1.0, "B": 2.5}
    served = {"A": 1.0, "B": 2.0, "C": 3.0}

    posted = changed_outputs(out, served, frozenset({"C"}), frozenset(), frozenset())

    assert list(posted) == ["C", "B"]


def test_a_none_value_passes_through() -> None:
    posted = changed_outputs({"A": None}, {"A": None}, frozenset(), frozenset(), frozenset())
    assert posted == {"A": None}


# -- timed passes --------------------------------------------------------------

FAST = frozenset({"I:CURRENT:FAST"})
EVENT = frozenset({"I:COUNT", "I:CHARGE"})
TIMED = FAST | EVENT
NONE: frozenset[str] = frozenset()


def test_a_timed_pass_reads_the_fast_names_and_the_emitted_ones() -> None:
    assert timed_pass_names(FAST, {"I:COUNT"}) == {"I:CURRENT:FAST", "I:COUNT"}


def test_a_timed_pass_with_nothing_emitted_reads_the_fast_names_alone() -> None:
    assert timed_pass_names(FAST, NONE) == FAST


def test_a_view_with_neither_class_reads_nothing_on_a_timed_pass() -> None:
    assert timed_pass_names(NONE, NONE) == frozenset()
    assert forced_names(NONE, NONE, NONE) == frozenset()


def test_a_timed_pass_forces_the_fast_and_emitted_names() -> None:
    assert forced_names(FAST, {"I:COUNT"}, NONE) == {"I:CURRENT:FAST", "I:COUNT"}


@pytest.mark.parametrize("faulted", ["I:CURRENT:FAST", "I:COUNT"], ids=["fast", "emitted"])
def test_a_read_faulted_name_is_never_forced(faulted: str) -> None:
    forced = forced_names(FAST, {"I:COUNT"}, {faulted})

    assert faulted not in forced
    assert forced == {"I:CURRENT:FAST", "I:COUNT"} - {faulted}


def test_a_pass_that_is_not_timed_forces_only_what_it_drained() -> None:
    assert forced_names(NONE, {"I:CHARGE"}, NONE) == {"I:CHARGE"}
    assert forced_names(NONE, NONE, NONE) == frozenset()


def test_a_pass_with_no_inputs_posts_no_unchanged_fast_or_event_name() -> None:
    out = {"I:CURRENT:FAST": 5.0, "I:COUNT": 3, "I:CURRENT": 5.0}
    served = dict(out)

    posted = untimed_outputs(out, served, TIMED, NONE, NONE, NONE)

    assert posted == {"I:CURRENT": 5.0}


def test_a_global_tick_posts_no_unchanged_name() -> None:
    # A tick reads every served channel; a client already holds all but what
    # moved since the last post.
    out = {"I:CURRENT:FAST": 5.0, "I:COUNT": 3, "I:CURRENT": 5.0, "M:TUNE:X": 0.25}
    served = {**out, "M:TUNE:X": 0.2}

    posted = untimed_outputs(out, served, TIMED, NONE, NONE, NONE, tick=True)

    assert posted == {"M:TUNE:X": 0.25}


def test_a_global_tick_posts_a_name_never_served_and_one_whose_undefined_condition_flipped() -> (
    None
):
    out = {"M:BPM:X": 1.0, "M:BPM:Y": 2.0, "M:TUNE:X": 0.25}
    served = {"M:BPM:X": 1.0, "M:TUNE:X": 0.25}

    posted = untimed_outputs(out, served, NONE, NONE, {"M:TUNE:X"}, NONE, tick=True)

    assert posted == {"M:BPM:Y": 2.0, "M:TUNE:X": 0.25}


def test_a_global_tick_posts_a_changed_fast_or_event_name() -> None:
    out = {"I:CURRENT:FAST": 6.0, "I:COUNT": 4, "I:CURRENT": 5.0}
    served = {"I:CURRENT:FAST": 5.0, "I:COUNT": 3, "I:CURRENT": 5.0}

    posted = untimed_outputs(out, served, TIMED, NONE, NONE, NONE, tick=True)

    assert list(posted) == ["I:CURRENT:FAST", "I:COUNT"]


def test_a_fault_transition_of_an_event_name_is_posted_on_a_global_tick() -> None:
    out = {"I:CHARGE": 1.0}

    posted = untimed_outputs(out, {"I:CHARGE": 1.0}, TIMED, NONE, {"I:CHARGE"}, NONE, tick=True)

    assert posted == {"I:CHARGE": 1.0}


def test_the_start_up_pass_posts_every_name_never_served() -> None:
    out = {"I:CURRENT:FAST": 5.0, "I:COUNT": 0, "I:CURRENT": 5.0}

    assert untimed_outputs(out, {}, TIMED, NONE, NONE, NONE) == out


def test_a_view_with_neither_class_posts_everything_on_a_pass_with_no_inputs() -> None:
    out = {"M:BPM:X": 1.0, "M:TUNE:X": 0.25}

    assert untimed_outputs(out, dict(out), NONE, NONE, NONE, NONE) == out


def test_a_write_pass_without_timed_names_posts_what_changed_outputs_posts() -> None:
    # The write pass's roster already leaves the periodic names out; the rule
    # posts nothing beyond it and narrows it as the write pass always has.
    periodic = {"M:CHROM:X"}
    roster = {"M:HCM:SP": 1.0, "M:BPM:X": 2.0, "M:TUNE:X": 0.25}
    served = {"M:HCM:SP": 1.0, "M:BPM:X": 2.0, "M:TUNE:X": 0.2, "M:CHROM:X": 1.5}
    written = {"M:HCM:SP"}

    posted = untimed_outputs(roster, served, NONE, NONE, NONE, NONE, written)

    assert posted == changed_outputs(roster, served, written, NONE, NONE)
    assert posted == {"M:HCM:SP": 1.0, "M:TUNE:X": 0.25}
    assert periodic.isdisjoint(posted)


def test_a_write_pass_posts_fast_and_event_names_only_when_changed() -> None:
    roster = {"I:REQUEST:SP": 1, "I:CURRENT:FAST": 5.0, "I:COUNT": 3, "I:CURRENT": 5.0}
    served = {"I:REQUEST:SP": 1, "I:CURRENT:FAST": 5.0, "I:COUNT": 3, "I:CURRENT": 4.0}

    posted = untimed_outputs(roster, served, TIMED, NONE, NONE, NONE, {"I:REQUEST:SP"})

    assert posted == {"I:REQUEST:SP": 1, "I:CURRENT": 5.0}


def test_an_emitted_name_on_a_pass_that_is_not_timed_is_posted_and_forced_once() -> None:
    # A step falls due during a global tick at a repeated value: that tick
    # publishes it, and the next timed pass, with the emitted set drained,
    # neither reads nor forces it again.
    out = {"I:CURRENT:FAST": 5.0, "I:COUNT": 3, "I:CURRENT": 5.0}
    served = dict(out)
    emitted = {"I:COUNT"}

    posted = untimed_outputs(out, served, TIMED, emitted, NONE, NONE, tick=True)
    forced = forced_names(NONE, emitted, NONE)

    assert posted == {"I:COUNT": 3}
    assert forced == {"I:COUNT"}

    next_read = timed_pass_names(FAST, NONE)
    assert "I:COUNT" not in next_read
    assert "I:COUNT" not in forced_names(FAST, NONE, NONE)


def test_an_emitted_name_on_a_write_pass_is_posted_unchanged() -> None:
    roster = {"I:REQUEST:SP": 1, "I:COUNT": 3}
    served = dict(roster)

    posted = untimed_outputs(roster, served, TIMED, {"I:COUNT"}, NONE, NONE, {"I:REQUEST:SP"})

    assert posted == roster


def test_an_emitted_name_the_pass_did_not_read_is_not_invented() -> None:
    posted = untimed_outputs({"I:CURRENT": 5.0}, {}, TIMED, {"I:COUNT"}, NONE, NONE)

    assert posted == {"I:CURRENT": 5.0}


def test_the_next_wake_is_a_fast_period_on_without_a_deadline() -> None:
    assert next_wake(100.0, 0.1) == pytest.approx(100.1)
    assert next_wake(100.0, 0.1, None) == pytest.approx(100.1)


def test_the_next_wake_is_an_earlier_deadline() -> None:
    assert next_wake(100.0, 0.1, 100.05) == 100.05


def test_a_later_deadline_waits_for_the_fast_period() -> None:
    assert next_wake(100.0, 0.1, 100.5) == pytest.approx(100.1)


def test_a_past_deadline_is_due_at_once() -> None:
    assert next_wake(100.0, 0.1, 99.0) == 99.0


@pytest.mark.parametrize("period", [0.0, -0.1, math.nan], ids=["zero", "negative", "nan"])
def test_a_fast_period_not_above_zero_is_refused(period: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        next_wake(100.0, period)


def test_the_module_imports_no_server_library() -> None:
    tree = ast.parse(Path(publication.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert roots.isdisjoint({"lume_pva_apg", "pcaspy", "p4p"})


# -- the built composite -------------------------------------------------------


class _Clock:
    """A fixed instant the test advances by hand."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _undefined(composite: Any, names: list[str]) -> frozenset[str]:
    return frozenset(
        name
        for name, entry in composite.output_severity(names).items()
        if entry.get("condition") == "udf"
    )


def test_publishing_only_changes_serves_what_a_full_publish_serves(
    built_demo: BuiltProject,
) -> None:
    pytest.importorskip("at")
    from osprey_connectors.simulation.composite import Composite
    from osprey_connectors.simulation.view import SimulatorView

    view = SimulatorView.of_render(built_demo.build_dir)
    clock = _Clock(1_760_000_000.0)
    composite = Composite(view, model_log=False, clock=clock)
    periodic = periodic_addresses(view)
    roster = [name for name in composite.supported_variables if name not in periodic]
    correctors = sorted(
        binding.address
        for binding in view.bindings()
        if binding.engine_role == "setpoint"
        and binding.direction == "write"
        and binding.plane is not None
        and view.channel(binding.address).writable
        and view.channel(binding.address).value_type == "float"
    )
    assert correctors, "the view states no writable float corrector setpoint"
    corrector = correctors[0]
    low, high = view.channel(corrector).value_range or (-1.0, 1.0)

    served: dict[str, Any] = dict(composite.get(roster))
    udf_served = _undefined(composite, roster)
    for step in range(10):
        clock.now += 0.5
        composite.set({corrector: low + (high - low) * (step + 1) / 12})
        out = composite.get(roster)
        udf_now = _undefined(composite, roster)

        posted = changed_outputs(out, served, frozenset({corrector}), udf_now, udf_served)
        served.update(posted)
        udf_served = (udf_served - set(posted)) | (udf_now & set(posted))

        different = [name for name in roster if not same_value(served[name], out[name])]
        assert different == [], f"pass {step}: a client would hold stale {different[:5]}"
        assert udf_served == udf_now
        assert corrector in posted
        assert len(posted) < len(out), f"pass {step} posted every channel it read"
