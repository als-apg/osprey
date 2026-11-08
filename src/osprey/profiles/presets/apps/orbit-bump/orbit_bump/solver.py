"""The bump solver: pure functions over ring-order positions, no I/O.

A BPM position ``b`` is the b-th BPM of one plane in lattice order and a corrector
position ``c`` the c-th corrector of that plane; both wrap at the ring end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

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


@dataclass(frozen=True)
class SolveResult:
    """Corrector deltas, the predicted orbit at every BPM, and the singular values used."""

    delta: np.ndarray
    predicted: np.ndarray
    sigma_kept: int
    sigma_total: int


def solve_bump(
    orm: np.ndarray,
    now: np.ndarray,
    goal: np.ndarray,
    bump_rows: list[int],
    leak_rows: list[int],
    sigma_cut: float = 1e-4,
) -> SolveResult:
    """Solve the corrector deltas that move ``bump_rows`` to ``goal`` and hold ``leak_rows``.

    ``orm`` is the response of every BPM of the plane (rows) to each selected corrector
    (columns); ``now`` and ``goal`` are orbits at every BPM. The fit uses the bump rows,
    aimed at ``goal``, and the leakage rows, aimed at ``now``; span rows that are not
    bump rows stay free. Bump rows weigh ``0.5 · n_leak / n_bump``, leakage rows 1.
    ``delta = V Σ⁻¹ Uᵀ W (goal − now)`` from the SVD of ``W·R``, keeping the singular
    values above ``sigma_cut`` times the largest. ``predicted = now + orm · delta``.

    Raises:
        ValueError: on no bump row, no leakage row, mismatched shapes, or a delta that
            is not finite.
    """
    orm = np.asarray(orm, dtype=float)
    now = np.asarray(now, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if orm.ndim != 2:
        raise ValueError(f"the response matrix must be 2-D, not shape {orm.shape}")
    n_bpm = orm.shape[0]
    if now.shape != (n_bpm,) or goal.shape != (n_bpm,):
        raise ValueError(
            f"orbit shape mismatch: the response matrix has {n_bpm} BPM rows, "
            f"now has shape {now.shape} and goal {goal.shape}"
        )
    if not bump_rows:
        raise ValueError("a bump needs at least one bump row")
    if not leak_rows:
        raise ValueError("no leakage BPM is left outside the bump to hold the orbit closed")
    rows = list(bump_rows) + list(leak_rows)
    weights = np.concatenate(
        [np.full(len(bump_rows), 0.5 * len(leak_rows) / len(bump_rows)), np.ones(len(leak_rows))]
    )
    target = np.concatenate([goal[bump_rows] - now[bump_rows], np.zeros(len(leak_rows))])
    weighted = weights[:, None] * orm[rows]
    u, s, vt = np.linalg.svd(weighted, full_matrices=False)
    keep = s > sigma_cut * s[0] if s.size else np.zeros(0, dtype=bool)
    delta = vt[keep].T @ ((u[:, keep].T @ (weights * target)) / s[keep])
    if not np.all(np.isfinite(delta)):
        raise ValueError("the solve gave a corrector delta that is not finite")
    return SolveResult(
        delta=delta,
        predicted=now + orm @ delta,
        sigma_kept=int(np.count_nonzero(keep)),
        sigma_total=int(s.size),
    )
