"""The bump solver: pure functions over ring-order positions, no I/O.

A BPM position ``b`` is the b-th BPM of one plane in lattice order and a corrector
position ``c`` the c-th corrector of that plane; both wrap at the ring end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from scipy.optimize import lsq_linear

#: The rule that picks one corrector upstream of the bump and two downstream.
AUTO = "auto"
#: The rule that takes the correctors the operator ticked.
MANUAL = "manual"
#: The MML signed-index lists the panel offers beside ``auto`` and ``manual``.
SIGNED_RULES = ("[-1 1]", "[-2 -1 1 2]", "[-3 -1 1 3]")
#: Every corrector rule, in the panel's dropdown order.
RULES = (AUTO, *SIGNED_RULES, MANUAL)

_SIGNED = re.compile(r"\[\s*-?\d+(?:[\s,]+-?\d+)*\s*\]")


@dataclass(frozen=True)
class SelectResult:
    """The correctors a rule picked, in arc order, and the unsolved ones it skipped."""

    selected: list[int]
    skipped_unsolved: list[int]


def select_correctors(
    bump_bpms: list[int],
    next_cor: list[int],
    n_cor: int,
    rule: str,
    manual: list[int] | None,
    unsolved: set[int],
) -> SelectResult:
    """Pick the correctors of a bump over ``bump_bpms`` by ``rule``.

    ``next_cor[b]`` is the first corrector downstream of BPM ``b``, wrapping at the
    ring end. ``auto`` takes the corrector just upstream of the first bump BPM, every
    corrector between the first and the last, and two downstream of the last. A signed
    list such as ``[-2 -1 1 2]`` (MML's form) takes, beside the correctors between,
    the k-th corrector upstream of the first bump BPM for each ``-k`` and the k-th
    downstream of the last for each ``+k``. ``manual`` takes ``manual`` as given.
    The first and last bump BPM are those of the shortest arc holding them all, so a
    bump may straddle the ring end. Every index is wrapped modulo ``n_cor``;
    ``selected`` is in arc order, most upstream first. A corrector in ``unsolved`` is
    left out of ``selected`` and listed in ``skipped_unsolved``.

    Raises:
        ValueError: on an unknown rule, a missing manual list, no or an unknown bump
            BPM, or a pick that would wrap the whole ring.
    """
    if n_cor < 1:
        raise ValueError("the ring has no corrector")
    if rule == MANUAL:
        if manual is None:
            raise ValueError("the manual rule needs the ticked correctors")
        picked = _arc_order(sorted({c % n_cor for c in manual}), n_cor)
    else:
        offsets = _offsets(rule)
        picked = _around_bump(bump_bpms, next_cor, n_cor, offsets)
    return SelectResult(
        selected=[c for c in picked if c not in unsolved],
        skipped_unsolved=[c for c in picked if c in unsolved],
    )


def _offsets(rule: str) -> tuple[list[int], list[int]]:
    """The upstream and downstream counts a non-manual rule names."""
    if rule == AUTO:
        return [1], [1, 2]
    if _SIGNED.fullmatch(rule):
        values = [int(v) for v in re.findall(r"-?\d+", rule)]
        if 0 not in values and len(set(values)) == len(values):
            up = sorted((-v for v in values if v < 0), reverse=True)
            down = sorted(v for v in values if v > 0)
            return up, down
    raise ValueError(f"unknown corrector rule {rule!r}; the rules: {', '.join(RULES)}")


def _around_bump(
    bump_bpms: list[int], next_cor: list[int], n_cor: int, offsets: tuple[list[int], list[int]]
) -> list[int]:
    """The correctors upstream, between and downstream of the bump, in arc order."""
    if not bump_bpms:
        raise ValueError("a bump needs at least one bump BPM")
    n_bpm = len(next_cor)
    unknown = sorted({b for b in bump_bpms if not 0 <= b < n_bpm})
    if unknown:
        raise ValueError(f"no BPM at ring position {unknown[0]}; the ring has {n_bpm}")
    arc = _arc_order(sorted(set(bump_bpms)), n_bpm)
    start = next_cor[arc[0]] % n_cor
    end = next_cor[arc[-1]] % n_cor
    between = (end - start) % n_cor
    up, down = offsets
    reach_up = max(up, default=0)
    reach_down = max(down, default=0)
    if reach_up + between + reach_down > n_cor:
        raise ValueError(
            f"the pick would wrap the whole ring: {reach_up} upstream, {between} between "
            f"and {reach_down} downstream of {n_cor} correctors"
        )
    picked = [start - k for k in up]
    picked += [start + i for i in range(between)]
    picked += [end + k - 1 for k in down]
    return [c % n_cor for c in picked]


def _arc_order(positions: list[int], n: int) -> list[int]:
    """Sorted ring ``positions`` reordered to start after the widest gap between them.

    The result is the shortest arc holding every position, most upstream first; on a
    tie the arc that does not cross the ring end wins.
    """
    if len(positions) < 2:
        return positions
    widest = n - positions[-1] + positions[0]
    start = 0
    for i in range(1, len(positions)):
        gap = positions[i] - positions[i - 1]
        if gap > widest:
            widest, start = gap, i
    return positions[start:] + positions[:start]


@dataclass(frozen=True)
class SplitResult:
    """BPM positions inside the bump's span and outside it (the leakage rows)."""

    span_rows: list[int]
    leak_rows: list[int]


def split_rows(bpm_idx: list[int], selected_cor_idx: list[int], n: int) -> SplitResult:
    """Split the plane's BPMs into span rows and leakage rows.

    ``bpm_idx[b]`` is the lattice index of BPM position ``b``, ``selected_cor_idx`` the
    lattice indices of the selected correctors in arc order (as ``select_correctors``
    returns them), and ``n`` the number of lattice elements in the ring. The span runs
    forward in ring order from ``selected_cor_idx[0]`` to ``selected_cor_idx[-1]``,
    wrapping at the ring end; a BPM inside it is a span row, every other BPM a leakage
    row. Both lists are BPM positions in ascending order.

    Raises:
        ValueError: on no selected corrector or a ring of no element.
    """
    if n < 1:
        raise ValueError("the ring has no element")
    if not selected_cor_idx:
        raise ValueError("a bump needs at least one selected corrector")
    start = selected_cor_idx[0] % n
    length = (selected_cor_idx[-1] - start) % n
    span: list[int] = []
    leak: list[int] = []
    for b, idx in enumerate(bpm_idx):
        (span if (idx - start) % n <= length else leak).append(b)
    return SplitResult(span_rows=span, leak_rows=leak)


#: Close the bump first: bump rows weigh ``0.5 · n_leak / n_bump`` against leakage rows.
CLOSURE: Final = "closure"
#: Hit the bump BPMs exactly when the correctors can, then hold the leakage rows.
TARGETS: Final = "targets"
#: The solve priorities, the default first.
PRIORITIES = (CLOSURE, TARGETS)
#: A bump row's weight against a leakage row when a limit turns the targets priority's
#: equality into a penalty.
TARGETS_WEIGHT = 1e4
#: How close to a limit, relative to the limit's size, a delta counts as at it.
_AT_LIMIT = 1e-9
#: How hard a bounded solve keeps the delta in the kept singular directions, relative
#: to the largest singular value of the weighted system.
_SUBSPACE_WEIGHT = 1e3


@dataclass(frozen=True)
class SolveResult:
    """Corrector deltas, the predicted orbit at every BPM, and the singular values.

    ``sigma_values`` is the spectrum the solve truncated, the kept ones first:
    ``sigma_kept`` of its ``sigma_total`` values. Under ``targets`` its first
    ``sigma_targets`` values are the bump step's and the rest the closure step's, each
    run descending; under ``closure`` ``sigma_targets`` is 0. ``bounded`` lists the
    columns whose delta sits at one of its limits.
    """

    delta: np.ndarray
    predicted: np.ndarray
    sigma_kept: int
    sigma_total: int
    bounded: list[int] = field(default_factory=list)
    sigma_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_targets: int = 0


@dataclass(frozen=True)
class _Truncated:
    """One truncated solve: the delta, an orthonormal basis of the kept corrector
    directions, the spectrum, how many of it are kept, and the bump step's share."""

    delta: np.ndarray
    basis: np.ndarray
    values: np.ndarray
    kept: int
    targets: int = 0


def solve_bump(
    orm: np.ndarray,
    now: np.ndarray,
    goal: np.ndarray,
    bump_rows: list[int],
    leak_rows: list[int],
    sigma_cut: float = 1e-4,
    *,
    sigma_keep: int | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    priority: str = CLOSURE,
) -> SolveResult:
    """Solve the corrector deltas that move ``bump_rows`` and ``leak_rows`` to ``goal``.

    ``orm`` is the response of every BPM of the plane (rows) to each selected corrector
    (columns); ``now`` and ``goal`` are orbits at every BPM. A caller holds the leakage
    rows by aiming them at ``now``; rows that are neither stay free. With no leakage row
    the bump rows are fitted alone, the smallest delta among equal fits.

    ``closure``: bump rows weigh ``0.5 · n_leak / n_bump``, leakage rows 1, and
    ``delta = V Σ⁻¹ Uᵀ W (goal − now)`` from the SVD of ``W·R``, keeping the singular
    values above ``sigma_cut`` times the largest. ``targets``: the bump rows are met
    exactly when the columns can, and the leakage rows are fitted in what freedom is
    left; each step keeps its singular values above ``sigma_cut`` times its largest.

    ``sigma_keep`` replaces the cut with a count: the ``sigma_keep`` largest non-zero
    singular values, clamped to the spectrum. Under ``targets`` the count spends the
    bump step's directions first and the closure step's after, so a smaller count
    drops closure directions before any target direction.

    ``lower`` and ``upper`` bound each delta (``±inf`` or ``None`` for none). When the
    truncated delta breaks one, the same weighted system is solved as a bounded least
    squares (``targets`` weighs its bump rows ``TARGETS_WEIGHT``), held to the kept
    directions; ``bounded`` names the columns left at a limit.
    ``predicted = now + orm · delta``.

    Raises:
        ValueError: on no bump row, mismatched shapes, an unknown priority, a
            ``sigma_keep`` below 1, or a delta that is not finite.
    """
    orm = np.asarray(orm, dtype=float)
    now = np.asarray(now, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if orm.ndim != 2:
        raise ValueError(f"the response matrix must be 2-D, not shape {orm.shape}")
    n_bpm, n_cor = orm.shape
    if now.shape != (n_bpm,) or goal.shape != (n_bpm,):
        raise ValueError(
            f"orbit shape mismatch: the response matrix has {n_bpm} BPM rows, "
            f"now has shape {now.shape} and goal {goal.shape}"
        )
    if priority not in PRIORITIES:
        raise ValueError(f"unknown priority {priority!r}; one of {', '.join(PRIORITIES)}")
    if not bump_rows:
        raise ValueError("a bump needs at least one bump row")
    if sigma_keep is not None and sigma_keep < 1:
        raise ValueError(f"sigma_keep must be at least 1, not {sigma_keep}")
    lo, hi = _limits(lower, upper, n_cor)
    rows = list(bump_rows) + list(leak_rows)
    target = goal[rows] - now[rows]
    if priority == CLOSURE:
        bump_weight = 0.5 * len(leak_rows) / len(bump_rows) if leak_rows else 1.0
        solved = _svd_solve(
            np.concatenate([np.full(len(bump_rows), bump_weight), np.ones(len(leak_rows))]),
            orm[rows],
            target,
            sigma_cut,
            sigma_keep,
        )
    else:
        bump_weight = TARGETS_WEIGHT
        solved = _exact_then_closed(
            orm[bump_rows],
            target[: len(bump_rows)],
            orm[leak_rows],
            target[len(bump_rows) :],
            sigma_cut,
            sigma_keep,
        )
    delta = solved.delta
    bounded: list[int] = []
    if np.any(delta < lo) or np.any(delta > hi):
        weights = np.concatenate([np.full(len(bump_rows), bump_weight), np.ones(len(leak_rows))])
        delta = _bounded_solve(weights[:, None] * orm[rows], weights * target, lo, hi, solved.basis)
        bounded = _at_limit(delta, lo, hi)
    if not np.all(np.isfinite(delta)):
        raise ValueError("the solve gave a corrector delta that is not finite")
    return SolveResult(
        delta=delta,
        predicted=now + orm @ delta,
        sigma_kept=solved.kept,
        sigma_total=int(solved.values.size),
        bounded=bounded,
        sigma_values=solved.values,
        sigma_targets=solved.targets,
    )


def _limits(
    lower: np.ndarray | None, upper: np.ndarray | None, n_cor: int
) -> tuple[np.ndarray, np.ndarray]:
    """The delta bounds as two arrays, open where none is given."""
    lo = np.full(n_cor, -np.inf) if lower is None else np.asarray(lower, dtype=float)
    hi = np.full(n_cor, np.inf) if upper is None else np.asarray(upper, dtype=float)
    if lo.shape != (n_cor,) or hi.shape != (n_cor,):
        raise ValueError(
            f"bound shape mismatch: {n_cor} correctors, lower has shape {lo.shape} "
            f"and upper {hi.shape}"
        )
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("a corrector's lower bound is above its upper bound or not a number")
    return lo, hi


def _kept(s: np.ndarray, sigma_cut: float, sigma_keep: int | None) -> int:
    """How many of the descending singular values ``s`` a solve keeps: those above
    ``sigma_cut`` times the largest, or the ``sigma_keep`` largest that are not zero."""
    if not s.size:
        return 0
    if sigma_keep is None:
        return int(np.count_nonzero(s > sigma_cut * s[0]))
    return min(sigma_keep, int(np.count_nonzero(s > 0)))


def _svd_solve(
    weights: np.ndarray,
    matrix: np.ndarray,
    target: np.ndarray,
    sigma_cut: float,
    sigma_keep: int | None,
) -> _Truncated:
    """The truncated-SVD solution of ``W·matrix·delta = W·target``."""
    u, s, vt = np.linalg.svd(weights[:, None] * matrix, full_matrices=False)
    k = _kept(s, sigma_cut, sigma_keep)
    delta = vt[:k].T @ ((u[:, :k].T @ (weights * target)) / s[:k])
    return _Truncated(delta=delta, basis=vt[:k].T, values=s, kept=k)


def _exact_then_closed(
    bump: np.ndarray,
    bump_target: np.ndarray,
    leak: np.ndarray,
    leak_target: np.ndarray,
    sigma_cut: float,
    sigma_keep: int | None,
) -> _Truncated:
    """Meet the bump rows (closest fit when they cannot be met), then fit the leakage
    rows in the null space the bump rows leave.

    The spectrum is the bump step's values above the cut, then the closure step's in
    the directions those leave; ``sigma_keep`` counts through it in that order, and the
    closure step runs only once every bump direction is kept.
    """
    u, s, vt = np.linalg.svd(bump, full_matrices=True)
    rank = _kept(s, sigma_cut, None)
    first = rank if sigma_keep is None else min(sigma_keep, rank)
    delta = vt[:first].T @ ((u[:, :first].T @ bump_target) / s[:first])
    basis = vt[:first].T
    values = s[:rank]
    kept = first
    free = vt[rank:].T
    if free.shape[1] and leak.shape[0]:
        ul, sl, vlt = np.linalg.svd(leak @ free, full_matrices=False)
        values = np.concatenate([values, sl])
        rest = None if sigma_keep is None else max(0, sigma_keep - rank)
        second = _kept(sl, sigma_cut, rest) if first == rank else 0
        z = vlt[:second].T @ ((ul[:, :second].T @ (leak_target - leak @ delta)) / sl[:second])
        delta = delta + free @ z
        basis = np.hstack([basis, free @ vlt[:second].T])
        kept += second
    return _Truncated(delta=delta, basis=basis, values=values, kept=kept, targets=rank)


def _bounded_solve(
    matrix: np.ndarray, target: np.ndarray, lo: np.ndarray, hi: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """The bounded least-squares solution of ``matrix·delta = target``, held to the
    span of ``basis`` (orthonormal columns) by a stiff penalty on the rest.

    A column whose two bounds meet is fixed at them and left out of the fit.
    """
    n_cor = matrix.shape[1]
    if basis.shape[1] < n_cor and matrix.size:
        stiff = _SUBSPACE_WEIGHT * np.linalg.norm(matrix, 2)
        matrix = np.vstack([matrix, stiff * (np.eye(n_cor) - basis @ basis.T)])
        target = np.concatenate([target, np.zeros(n_cor)])
    fixed = hi - lo <= 0
    delta = np.where(fixed, lo, 0.0)
    loose = ~fixed
    if np.any(loose):
        rest = target - matrix[:, fixed] @ delta[fixed]
        found = lsq_linear(matrix[:, loose], rest, bounds=(lo[loose], hi[loose]), method="bvls")
        delta[loose] = found.x
    return delta


def _at_limit(delta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[int]:
    """The columns whose delta sits at one of its finite bounds."""
    hit = []
    for c, (d, low, high) in enumerate(zip(delta, lo, hi, strict=True)):
        for bound in (low, high):
            if np.isfinite(bound) and abs(d - bound) <= _AT_LIMIT * max(1.0, abs(bound)):
                hit.append(c)
                break
    return hit
