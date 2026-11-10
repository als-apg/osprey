"""The orbit-bump solver: which correctors a rule picks for a bump, and the solve.

Positions are ring-order positions: BPM ``b`` is the b-th BPM of the plane in lattice
order, corrector ``c`` the c-th corrector. ``next_cor[b]`` is the first corrector
downstream of BPM ``b``, wrapping at the ring end.
"""

import numpy as np
import pytest
from orbit_bump.solver import (
    SelectResult,
    SolveResult,
    SplitResult,
    select_correctors,
    solve_bump,
    split_rows,
)

# A non-alternating ring of 5 BPMs and 7 correctors, in lattice order:
#
#     C0 B0 C1 C2 B1 B2 C3 B3 C4 C5 C6 B4 | (wraps to C0)
#
# B1 and B2 share no corrector between them; B4's first downstream corrector is C0.
N_COR = 7
NEXT_COR = [1, 3, 3, 4, 0]


def _pick(bump, rule="auto", *, manual=None, unsolved=frozenset(), next_cor=NEXT_COR, n_cor=N_COR):
    return select_correctors(bump, next_cor, n_cor, rule, manual, set(unsolved))


def test_rules_auto_single_bpm_is_the_classic_three_bump():
    assert _pick([0]) == SelectResult(selected=[0, 1, 2], skipped_unsolved=[])
    assert _pick([1]).selected == [2, 3, 4]


def test_rules_auto_takes_every_corrector_between_the_bump_bpms():
    assert _pick([0, 2]).selected == [0, 1, 2, 3, 4]
    assert _pick([0, 1, 3]).selected == [0, 1, 2, 3, 4, 5]


def test_rules_auto_adjacent_bpms_without_a_corrector_between_on_a_non_alternating_ring():
    assert _pick([1, 2]).selected == [2, 3, 4]


def test_rules_auto_bump_order_given_does_not_matter():
    assert _pick([2, 0]).selected == _pick([0, 2]).selected


def test_rules_the_bump_is_the_shortest_arc_holding_its_bpms():
    # B3 -> B4 -> B0 is two BPM steps, B0 -> B3 three: the bump crosses the ring end.
    assert _pick([0, 3]).selected == [3, 4, 5, 6, 0, 1, 2]


def test_rules_auto_on_an_alternating_ring_n_adjacent_bpms_take_n_plus_two():
    # C0 B0 C1 B1 C2 B2 C3 B3 C4 B4 C5 B5
    next_cor = [1, 2, 3, 4, 5, 0]
    for n in range(1, 5):
        picked = _pick(list(range(n)), next_cor=next_cor, n_cor=6).selected
        assert picked == list(range(n + 2))


def test_rules_wrap_around_pick_at_index_zero():
    # B4 sits after C6: one upstream is C6, two downstream are C0 and C1.
    assert _pick([4]).selected == [6, 0, 1]
    # Over the ring end: B4 then B0 in arc order, whatever order they are given in.
    assert _pick([0, 4]).selected == [6, 0, 1, 2]
    assert _pick([4, 0]).selected == [6, 0, 1, 2]


def test_rules_signed_lists_count_from_the_first_and_last_bump_bpm():
    assert _pick([1, 3], "[-1 1]").selected == [2, 3, 4]
    assert _pick([1], "[-2 -1 1 2]").selected == [1, 2, 3, 4]
    # Upstream of B0 wraps: -3 from C1 is C5.
    assert _pick([0], "[-3 -1 1 3]").selected == [5, 0, 1, 3]


def test_rules_signed_lists_wrap_at_the_ring_end():
    assert _pick([4], "[-2 -1 1 2]").selected == [5, 6, 0, 1]


def test_rules_signed_lists_keep_the_correctors_between_the_bump_bpms():
    assert _pick([0, 2], "[-1 1]").selected == [0, 1, 2, 3]


def test_rules_manual_passes_through_wrapped_in_arc_order():
    assert _pick([], "manual", manual=[6, 0, 1]).selected == [6, 0, 1]
    assert _pick([], "manual", manual=[1, 0, 6]).selected == [6, 0, 1]
    assert _pick([], "manual", manual=[7, 2]).selected == [0, 2]
    assert _pick([], "manual", manual=[3, 3, 4]).selected == [3, 4]


def test_rules_unsolved_correctors_are_skipped_and_reported():
    result = _pick([0], unsolved={1})
    assert result == SelectResult(selected=[0, 2], skipped_unsolved=[1])
    assert _pick([4], unsolved={0, 6}) == SelectResult(selected=[1], skipped_unsolved=[6, 0])
    manual = _pick([], "manual", manual=[2, 3], unsolved={3})
    assert manual == SelectResult(selected=[2], skipped_unsolved=[3])


def test_rules_a_pick_that_wraps_the_whole_ring_is_refused():
    with pytest.raises(ValueError, match="whole ring"):
        _pick([0, 3], "[-3 -1 1 3]")
    with pytest.raises(ValueError, match="whole ring"):
        _pick([0, 1, 2, 3, 4])


@pytest.mark.parametrize(
    ("bump", "rule", "manual", "match"),
    [
        ([0], "nearest", None, "unknown corrector rule"),
        ([0], "[-1 0 1]", None, "unknown corrector rule"),
        ([0], "[-1 -1 1]", None, "unknown corrector rule"),
        ([], "auto", None, "at least one bump BPM"),
        ([5], "auto", None, "no BPM"),
        ([0], "manual", None, "manual"),
    ],
)
def test_rules_bad_requests_are_refused(bump, rule, manual, match):
    with pytest.raises(ValueError, match=match):
        _pick(bump, rule, manual=manual)


# --- split_rows and solve_bump ------------------------------------------------------
#
# The same ring in lattice indices (12 elements):
#
#     C0=0 B0=1 C1=2 C2=3 B1=4 B2=5 C3=6 B3=7 C4=8 C5=9 C6=10 B4=11
N_ELEM = 12
BPM_IDX = [1, 4, 5, 7, 11]
COR_IDX = [0, 2, 3, 6, 8, 9, 10]


def _split(bump, rule="auto"):
    selected = _pick(bump, rule).selected
    return split_rows(BPM_IDX, [COR_IDX[c] for c in selected], N_ELEM)


def test_solve_split_rows_inside_the_span_without_a_wrap():
    # [0] picks C0 C1 C2: lattice 0..3 holds only B0.
    assert _split([0]) == SplitResult(span_rows=[0], leak_rows=[1, 2, 3, 4])


def test_solve_split_rows_span_wrapping_past_the_ring_start():
    # [0, 3] picks C3..C6 C0 C1 C2: lattice 6 -> 11 -> 0 -> 3 holds B3, B4 and B0.
    assert _split([0, 3]) == SplitResult(span_rows=[0, 3, 4], leak_rows=[1, 2])


def test_solve_split_rows_pick_straddling_the_ring_end_leaves_leakage():
    # Bump BPMs {n_bpm - 1, 0}: C6 C0 C1 C2, lattice 10 -> 11 -> 0 -> 3.
    n_bpm = len(BPM_IDX)
    result = _split([n_bpm - 1, 0])
    assert result.span_rows == [0, n_bpm - 1]
    assert result.leak_rows == [1, 2, 3]


def test_solve_split_rows_walks_from_first_to_last_selected_not_sorted():
    # Same correctors, given out of arc order, would span the other way round.
    assert split_rows(BPM_IDX, [10, 3], N_ELEM) == SplitResult(
        span_rows=[0, 4], leak_rows=[1, 2, 3]
    )
    assert split_rows(BPM_IDX, [3, 10], N_ELEM) == SplitResult(
        span_rows=[1, 2, 3], leak_rows=[0, 4]
    )


def test_solve_split_rows_refuses_no_corrector():
    with pytest.raises(ValueError, match="corrector"):
        split_rows(BPM_IDX, [], N_ELEM)


def _closable_orm(n_bpm=8, bump_row=0, seed=1):
    """An ORM over 3 correctors whose non-bump rows admit a closed bump: column 2 is
    2·column 0 − column 1 on every row but ``bump_row``."""
    rng = np.random.default_rng(seed)
    orm = rng.normal(size=(n_bpm, 3))
    others = [r for r in range(n_bpm) if r != bump_row]
    orm[others, 2] = 2 * orm[others, 0] - orm[others, 1]
    return orm


def test_solve_three_bump_closes():
    orm = _closable_orm()
    now = np.linspace(-0.1, 0.2, 8)
    goal = now.copy()
    goal[0] += 0.5
    leak = list(range(1, 8))
    result = solve_bump(orm, now, goal, [0], leak)
    assert isinstance(result, SolveResult)
    assert result.delta.shape == (3,)
    assert result.predicted.shape == (8,)
    np.testing.assert_allclose(result.predicted[leak], now[leak], atol=1e-10)
    np.testing.assert_allclose(result.predicted[0], goal[0], atol=1e-10)
    np.testing.assert_allclose(result.predicted, now + orm @ result.delta)
    assert (result.sigma_kept, result.sigma_total) == (3, 3)


def test_solve_n_bpm_weights_match_lstsq():
    rng = np.random.default_rng(7)
    orm = rng.normal(size=(10, 5))
    now = rng.normal(size=10)
    goal = now.copy()
    bump = [2, 3, 4]
    leak = [0, 1, 6, 7, 8, 9]  # row 5 is a span row outside the fit
    goal[bump] += [0.3, -0.2, 0.4]
    goal[5] += 99.0  # ignored: neither a bump nor a leakage row
    result = solve_bump(orm, now, goal, bump, leak)

    w = np.diag([0.5 * len(leak) / len(bump)] * len(bump) + [1.0] * len(leak))
    rows = bump + leak
    b = np.concatenate([goal[bump] - now[bump], np.zeros(len(leak))])
    expected, *_ = np.linalg.lstsq(w @ orm[rows], w @ b, rcond=None)
    np.testing.assert_allclose(result.delta, expected, atol=1e-12)
    np.testing.assert_allclose(result.predicted, now + orm @ expected, atol=1e-12)
    assert (result.sigma_kept, result.sigma_total) == (5, 5)


def test_solve_sigma_cutoff_drops_a_degenerate_column():
    rng = np.random.default_rng(3)
    orm = rng.normal(size=(8, 3))
    orm[:, 2] = orm[:, 1]  # two correctors with the same response
    now = np.zeros(8)
    goal = np.zeros(8)
    goal[0] = 1.0
    result = solve_bump(orm, now, goal, [0], list(range(1, 8)))
    assert (result.sigma_kept, result.sigma_total) == (2, 3)
    assert np.all(np.isfinite(result.delta))
    # The minimum-norm answer shares the duplicated response evenly.
    assert result.delta[1] == pytest.approx(result.delta[2])


def test_solve_sigma_cut_is_relative_to_the_largest_singular_value():
    orm = np.diag([1.0, 1e-3, 1.0])
    orm = np.vstack([orm, [[1.0, 0.0, 1.0]]])  # column 1 stays orthogonal: sigma 1e-3
    now = np.zeros(4)
    goal = np.array([0.0, 0.0, 0.0, 1.0])
    kept_all = solve_bump(orm, now, goal, [3], [0, 1, 2], sigma_cut=1e-6)
    kept_two = solve_bump(orm, now, goal, [3], [0, 1, 2], sigma_cut=1e-2)
    assert kept_all.sigma_kept == 3
    assert kept_two.sigma_kept == 2


@pytest.mark.parametrize("priority", ["closure", "targets"])
def test_solve_without_leakage_rows_fits_the_bump_rows_alone(priority):
    orm = _closable_orm()
    goal = np.zeros(8)
    goal[[0, 1]] = [0.3, -0.2]
    result = solve_bump(orm, np.zeros(8), goal, [0, 1], [], priority=priority)
    assert result.predicted[[0, 1]] == pytest.approx([0.3, -0.2])
    # The smallest delta among the exact fits: no part of it leaves the bump rows unmoved.
    assert np.linalg.norm(result.delta) == pytest.approx(
        np.linalg.norm(np.linalg.pinv(orm[[0, 1]]) @ goal[[0, 1]])
    )


def test_solve_refuses_a_non_finite_delta():
    orm = _closable_orm()
    now = np.zeros(8)
    goal = np.zeros(8)
    goal[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_bump(orm, now, goal, [0], list(range(1, 8)))


def test_solve_refuses_mismatched_shapes_and_no_bump_row():
    orm = _closable_orm()
    with pytest.raises(ValueError, match="bump row"):
        solve_bump(orm, np.zeros(8), np.zeros(8), [], [1, 2])
    with pytest.raises(ValueError, match="shape"):
        solve_bump(orm, np.zeros(7), np.zeros(8), [0], [1, 2])


def test_solve_output_is_deterministic():
    orm = _closable_orm(seed=5)
    now = np.linspace(0, 1, 8)
    goal = now + np.r_[0.4, np.zeros(7)]
    first = solve_bump(orm, now, goal, [0], list(range(1, 8)))
    second = solve_bump(orm.copy(), now.copy(), goal.copy(), [0], list(range(1, 8)))
    assert np.array_equal(first.delta, second.delta)
    assert np.array_equal(first.predicted, second.predicted)
    assert (first.sigma_kept, first.sigma_total) == (second.sigma_kept, second.sigma_total)


# -- limits, priority, the aim of the leakage rows ----------------------------------


def _wide_problem(seed=11):
    """Ten BPMs, four correctors: three bump rows and six leakage rows, row 9 free."""
    rng = np.random.default_rng(seed)
    orm = rng.normal(size=(10, 4))
    now = rng.normal(scale=0.1, size=10)
    goal = now.copy()
    goal[[0, 1, 2]] += [0.4, 0.5, 0.3]
    return orm, now, goal, [0, 1, 2], [3, 4, 5, 6, 7, 8]


def test_solve_bounds_that_no_delta_reaches_leave_the_svd_solution_unchanged():
    orm, now, goal, bump, leak = _wide_problem()
    free = solve_bump(orm, now, goal, bump, leak)
    reach = np.abs(free.delta).max()
    held = solve_bump(
        orm, now, goal, bump, leak, lower=np.full(4, -2 * reach), upper=np.full(4, 2 * reach)
    )
    assert np.array_equal(held.delta, free.delta)
    assert np.array_equal(held.predicted, free.predicted)
    assert (held.sigma_kept, held.sigma_total) == (free.sigma_kept, free.sigma_total)
    assert held.bounded == [] == free.bounded


def test_solve_an_active_bound_is_respected_and_listed():
    orm, now, goal, bump, leak = _wide_problem()
    free = solve_bump(orm, now, goal, bump, leak)
    worst = int(np.argmax(np.abs(free.delta)))
    cap = 0.5 * abs(free.delta[worst])
    lower, upper = np.full(4, -np.inf), np.full(4, np.inf)
    lower[worst], upper[worst] = -cap, cap
    held = solve_bump(orm, now, goal, bump, leak, lower=lower, upper=upper)
    assert np.all(held.delta >= lower - 1e-12) and np.all(held.delta <= upper + 1e-12)
    assert held.bounded == [worst]
    assert abs(held.delta[worst]) == pytest.approx(cap)
    np.testing.assert_allclose(held.predicted, now + orm @ held.delta)
    # The bounded answer is the weighted least squares within the box: no box point fits better.
    w = np.r_[np.full(3, 0.5 * 6 / 3), np.ones(6)]
    rows = bump + leak
    target = goal[rows] - now[rows]

    def cost(d):
        return float(np.sum((w * (orm[rows] @ d - target)) ** 2))

    rng = np.random.default_rng(0)
    for _ in range(50):
        probe = np.clip(held.delta + rng.normal(scale=0.05, size=4), lower, upper)
        assert cost(held.delta) <= cost(probe) + 1e-12


def test_solve_a_bound_held_at_zero_width_fixes_that_delta():
    orm, now, goal, bump, leak = _wide_problem()
    lower, upper = np.full(4, -np.inf), np.full(4, np.inf)
    lower[1] = upper[1] = 0.0
    held = solve_bump(orm, now, goal, bump, leak, lower=lower, upper=upper)
    assert held.delta[1] == 0.0
    assert 1 in held.bounded


def test_solve_the_targets_priority_hits_targets_the_closure_priority_trades_away():
    orm, now, goal, bump, leak = _wide_problem()
    closure = solve_bump(orm, now, goal, bump, leak)
    targets = solve_bump(orm, now, goal, bump, leak, priority="targets")
    assert np.abs(closure.predicted[bump] - goal[bump]).max() > 1e-3
    np.testing.assert_allclose(targets.predicted[bump], goal[bump], atol=1e-12)
    # Among the exact solutions it holds the leakage rows best: one free direction is left.
    null = np.linalg.svd(orm[bump])[2][3:].T
    z, *_ = np.linalg.lstsq(
        orm[leak] @ null, goal[leak] - now[leak] - orm[leak] @ targets.delta, rcond=None
    )
    np.testing.assert_allclose(z, 0.0, atol=1e-12)
    assert (targets.sigma_kept, targets.sigma_total) == (4, 4)


def test_solve_the_targets_priority_falls_back_to_the_closest_fit():
    orm = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 1.0]])
    now = np.zeros(4)
    goal = np.array([1.0, 3.0, 0.0, 0.0])  # rows 0 and 1 see column 0 alike: no exact fit
    result = solve_bump(orm, now, goal, [0, 1], [2, 3], priority="targets")
    assert result.delta[0] == pytest.approx(2.0)
    assert np.all(np.isfinite(result.delta))


def test_solve_the_targets_priority_respects_a_bound():
    orm, now, goal, bump, leak = _wide_problem()
    free = solve_bump(orm, now, goal, bump, leak, priority="targets")
    cap = 0.5 * np.abs(free.delta).max()
    held = solve_bump(
        orm,
        now,
        goal,
        bump,
        leak,
        lower=np.full(4, -cap),
        upper=np.full(4, cap),
        priority="targets",
    )
    assert np.all(np.abs(held.delta) <= cap + 1e-12)
    assert held.bounded


def test_solve_leakage_rows_aim_at_goal():
    orm = _closable_orm()
    now = np.zeros(8)
    # Aim every row, leakage rows too, at an orbit the three correctors can reach.
    goal = orm @ np.array([0.1, -0.2, 0.05])
    result = solve_bump(orm, now, goal, [0], list(range(1, 8)))
    np.testing.assert_allclose(result.predicted, goal, atol=1e-10)


def test_solve_refuses_an_unknown_priority_and_mismatched_bounds():
    orm = _closable_orm()
    with pytest.raises(ValueError, match="priority"):
        solve_bump(orm, np.zeros(8), np.ones(8), [0], [1, 2], priority="both")
    with pytest.raises(ValueError, match="bound"):
        solve_bump(orm, np.zeros(8), np.ones(8), [0], [1, 2], lower=np.zeros(2))
    with pytest.raises(ValueError, match="bound"):
        solve_bump(orm, np.zeros(8), np.ones(8), [0], [1, 2], lower=np.ones(3), upper=np.zeros(3))


# -- the singular-value count ---------------------------------------------------------


def test_solve_reports_the_spectrum_of_the_weighted_system_descending():
    orm, now, goal, bump, leak = _wide_problem()
    result = solve_bump(orm, now, goal, bump, leak)
    w = np.r_[np.full(3, 0.5 * 6 / 3), np.ones(6)]
    expected = np.linalg.svd(w[:, None] * orm[bump + leak], compute_uv=False)
    np.testing.assert_allclose(result.sigma_values, expected)
    assert result.sigma_total == 4 and result.sigma_targets == 0


@pytest.mark.parametrize("priority", ["closure", "targets"])
def test_solve_a_count_equal_to_the_cuts_reproduces_the_cut(priority):
    orm, now, goal, bump, leak = _wide_problem()
    cut = solve_bump(orm, now, goal, bump, leak, priority=priority)
    counted = solve_bump(orm, now, goal, bump, leak, priority=priority, sigma_keep=cut.sigma_kept)
    np.testing.assert_allclose(counted.delta, cut.delta, atol=1e-12)
    assert counted.sigma_kept == cut.sigma_kept


def test_solve_a_count_truncates_the_closure_solve_to_its_largest_values():
    orm, now, goal, bump, leak = _wide_problem()
    result = solve_bump(orm, now, goal, bump, leak, sigma_keep=2)
    w = np.r_[np.full(3, 0.5 * 6 / 3), np.ones(6)]
    rows = bump + leak
    u, s, vt = np.linalg.svd(w[:, None] * orm[rows], full_matrices=False)
    expected = vt[:2].T @ ((u[:, :2].T @ (w * (goal[rows] - now[rows]))) / s[:2])
    np.testing.assert_allclose(result.delta, expected, atol=1e-12)
    assert (result.sigma_kept, result.sigma_total) == (2, 4)


@pytest.mark.parametrize("priority", ["closure", "targets"])
def test_solve_a_count_past_the_spectrum_is_clamped(priority):
    orm, now, goal, bump, leak = _wide_problem()
    every = solve_bump(orm, now, goal, bump, leak, priority=priority, sigma_keep=4)
    past = solve_bump(orm, now, goal, bump, leak, priority=priority, sigma_keep=99)
    np.testing.assert_array_equal(past.delta, every.delta)
    assert past.sigma_kept == past.sigma_total == 4


def test_solve_refuses_a_count_below_one():
    orm, now, goal, bump, leak = _wide_problem()
    with pytest.raises(ValueError, match="sigma_keep"):
        solve_bump(orm, now, goal, bump, leak, sigma_keep=0)


def test_solve_the_targets_spectrum_is_the_bump_step_then_the_closure_step():
    orm, now, goal, bump, leak = _wide_problem()
    result = solve_bump(orm, now, goal, bump, leak, priority="targets")
    _u, s, vt = np.linalg.svd(orm[bump], full_matrices=True)
    closure = np.linalg.svd(orm[leak] @ vt[3:].T, compute_uv=False)
    np.testing.assert_allclose(result.sigma_values, np.r_[s, closure])
    assert result.sigma_targets == 3


def test_solve_a_targets_count_drops_closure_before_any_target():
    orm, now, goal, bump, leak = _wide_problem()
    three = solve_bump(orm, now, goal, bump, leak, priority="targets", sigma_keep=3)
    # The bump rows are still met exactly; only the closure direction is gone.
    np.testing.assert_allclose(three.predicted[bump], goal[bump], atol=1e-12)
    np.testing.assert_allclose(three.delta, np.linalg.pinv(orm[bump]) @ (goal - now)[bump])
    two = solve_bump(orm, now, goal, bump, leak, priority="targets", sigma_keep=2)
    assert np.abs(two.predicted[bump] - goal[bump]).max() > 1e-3
    assert (two.sigma_kept, three.sigma_kept) == (2, 3)


@pytest.mark.parametrize("priority", ["closure", "targets"])
def test_solve_a_bounded_solve_stays_in_the_kept_directions(priority):
    orm, now, goal, bump, leak = _wide_problem()
    free = solve_bump(orm, now, goal, bump, leak, priority=priority, sigma_keep=2)
    cap = 0.5 * np.abs(free.delta).max()
    held = solve_bump(
        orm,
        now,
        goal,
        bump,
        leak,
        priority=priority,
        sigma_keep=2,
        lower=np.full(4, -cap),
        upper=np.full(4, cap),
    )
    assert np.all(np.abs(held.delta) <= cap + 1e-12)
    assert held.bounded and held.sigma_kept == 2
    # The delta lies in the span the unbounded truncated solve used.
    if priority == "closure":
        w = np.r_[np.full(3, 0.5 * 6 / 3), np.ones(6)]
        vt = np.linalg.svd(w[:, None] * orm[bump + leak], full_matrices=False)[2]
    else:
        vt = np.linalg.svd(orm[bump], full_matrices=True)[2]
    outside = np.eye(4) - vt[:2].T @ vt[:2]
    assert np.linalg.norm(outside @ held.delta) <= 1e-4 * np.linalg.norm(held.delta)
