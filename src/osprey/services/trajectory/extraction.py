"""Contiguous first-turn extraction of one shot, and averaging over shots.

Pure functions over turn-by-turn records already in rotated BPM order: row
``b`` of every array is BPM ``b``, column ``k`` is sample ``k``.

Every level is relative to the shot's **injected-charge scale**: the largest
single-sample rise of any BPM's sum, ``max(diff(sum))`` over every BPM and
sample, which is the charge the injection brought to the BPM it reached best.
A stored-beam baseline, an attenuated sum or a facility's own sum units shift
or stretch every rise alike, so the fractions mean the same everywhere.

A BPM *rises* when ``max(diff(sum))`` exceeds ``loss_cutoff`` times the scale;
a NaN difference never counts. The loss mask is contiguous: the first BPM that
does not rise and every BPM after it are lost, even when a later BPM rises
again. The first sample with beam at a rising BPM is ``argmax(diff(sum)) + 1``,
and the shot's ``i0`` is the lower median of that over the rising BPMs before
the loss. The first turn is sample ``i0``; the raw window is
``[max(i0 - trigger_sample, 0), min(i0 + raw_window_turns, n_samples))``, where
``trigger_sample`` is the declared sample every turn series' injected shot
arrives at, so the window keeps as many samples before ``i0`` as the buffer
holds before the trigger.

A shot fails when no BPM's sum rises at all, when no BPM rises before the loss
(the first BPM does not rise), or when the first BPM's first-turn level — its
sum at ``i0`` less its median before ``i0`` — is below ``shot_threshold`` times
the scale.

x and y are NaN, and the sum is 0, at every lost BPM: a BPM past the loss may
read finite values with no beam, and none of them survive. Averaging takes the
earliest loss over the included shots as the lost set and masks it again
after the mean, so a shot that kept beam further cannot leak values past it.
"""

from __future__ import annotations

import math
import statistics
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AVERAGE_MODES",
    "AveragedShot",
    "ExtractionFailure",
    "ShotExtraction",
    "average_shots",
    "extract_shot",
]

#: How included shots are combined per BPM.
AVERAGE_MODES = ("mean", "nanmean")

#: The reason a shot whose first BPM does not rise is excluded.
NO_RISE_REASON = "no BPM rose"


@dataclass(frozen=True, eq=False)
class ShotExtraction:
    """One included shot's first turn.

    Attributes:
        x: Horizontal first-turn position per BPM; NaN from ``lost_at`` on.
        y: Vertical first-turn position per BPM; NaN from ``lost_at`` on.
        sum: First-turn sum per BPM; 0 from ``lost_at`` on.
        lost_at: Index of the first lost BPM, ``None`` for a full turn.
        i0: The first sample with beam.
        window: The raw window ``(start, stop)`` in samples.
    """

    x: np.ndarray
    y: np.ndarray
    sum: np.ndarray
    lost_at: int | None
    i0: int
    window: tuple[int, int]

    @property
    def last_bpm(self) -> int:
        """Index of the last BPM with beam."""
        return _last_bpm(self.lost_at, len(self.x))


@dataclass(frozen=True)
class ExtractionFailure:
    """One shot excluded from the average, never retried.

    Attributes:
        reason: Why the shot failed.
        sum_level: The first BPM's first-turn level as a fraction of the
            injected-charge scale, ``None`` when no BPM rose.
    """

    reason: str
    sum_level: float | None = None


@dataclass(frozen=True, eq=False)
class AveragedShot:
    """The included shots of one shot action, combined.

    Attributes:
        x: Averaged horizontal first turn; NaN from ``lost_at`` on.
        y: Averaged vertical first turn; NaN from ``lost_at`` on.
        sum: Averaged first-turn sum; 0 from ``lost_at`` on.
        lost_at: The earliest first lost BPM over the shots, ``None`` when
            every shot made a full turn.
        i0: The lower median of the shots' ``i0``.
        window: The raw window of a shot whose ``i0`` is the median.
        per_shot_lost_at: Each shot's own first lost BPM, in shot order.
        shots_included: How many shots were averaged.
    """

    x: np.ndarray
    y: np.ndarray
    sum: np.ndarray
    lost_at: int | None
    i0: int
    window: tuple[int, int]
    per_shot_lost_at: list[int | None]
    shots_included: int

    @property
    def last_bpm(self) -> int:
        """Index of the last BPM with beam."""
        return _last_bpm(self.lost_at, len(self.x))


def _last_bpm(lost_at: int | None, n_bpm: int) -> int:
    return n_bpm - 1 if lost_at is None else lost_at - 1


def _mask(x: np.ndarray, y: np.ndarray, total: np.ndarray, lost_at: int | None) -> None:
    if lost_at is not None:
        x[lost_at:] = np.nan
        y[lost_at:] = np.nan
        total[lost_at:] = 0.0


def extract_shot(
    x: np.ndarray,
    y: np.ndarray,
    sum: np.ndarray,
    *,
    loss_cutoff: float,
    shot_threshold: float,
    raw_window_turns: int,
    trigger_sample: int,
) -> ShotExtraction | ExtractionFailure:
    """Extract one shot's first turn, or say why the shot fails.

    Args:
        x: Horizontal turn-by-turn records, shape ``(n_bpm, n_samples)``.
        y: Vertical turn-by-turn records, same shape.
        sum: Sum turn-by-turn records, same shape.
        loss_cutoff: A BPM rises when ``max(diff(sum))`` exceeds this fraction
            of the injected-charge scale.
        shot_threshold: The shot fails when the first BPM's first-turn level
            is below this fraction of the injected-charge scale.
        raw_window_turns: Samples kept from ``i0`` on in the raw window.
        trigger_sample: Samples kept before ``i0`` in the raw window: the
            declared sample the injected shot arrives at.

    Returns:
        The extraction, or the failure with its reason.

    Raises:
        ValueError: When the records are not three equal 2-D arrays with at
            least one BPM and two samples, or ``raw_window_turns`` or
            ``trigger_sample`` is negative.
    """
    xs, ys, sums = (np.asarray(a, dtype=float) for a in (x, y, sum))
    if xs.ndim != 2 or xs.shape != ys.shape or xs.shape != sums.shape:
        raise ValueError(
            f"x, y and sum must be 2-D arrays of one shape; got {xs.shape}, {ys.shape}, {sums.shape}"
        )
    n_bpm, n_samples = sums.shape
    if n_bpm < 1 or n_samples < 2:
        raise ValueError(f"records need at least one BPM and two samples; got shape {sums.shape}")
    if raw_window_turns < 0:
        raise ValueError(f"raw_window_turns must not be negative; got {raw_window_turns}")
    if trigger_sample < 0:
        raise ValueError(f"trigger_sample must not be negative; got {trigger_sample}")

    steps = np.diff(sums, axis=1)
    steps = np.where(np.isfinite(steps), steps, -np.inf)
    peaks = steps.max(axis=1)
    scale = float(peaks.max())
    if not scale > 0:
        return ExtractionFailure(reason=NO_RISE_REASON)
    rises = peaks > loss_cutoff * scale
    not_rising = np.flatnonzero(~rises)
    lost_at = int(not_rising[0]) if not_rising.size else None
    risers = n_bpm if lost_at is None else lost_at
    if risers == 0:
        return ExtractionFailure(reason=NO_RISE_REASON)

    starts = steps[:risers].argmax(axis=1) + 1
    i0 = int(statistics.median_low(starts.tolist()))
    with warnings.catch_warnings():
        # A first BPM with no finite sample before i0 has no baseline; it counts as 0.
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = float(np.nanmedian(sums[0, :i0]))
    if not math.isfinite(baseline):
        baseline = 0.0
    level = float(sums[0, i0] - baseline) / scale
    if not level >= shot_threshold:
        return ExtractionFailure(
            reason=(
                f"first-turn level {level:.3g} of the injected-charge scale at the first BPM "
                f"is below shot_threshold_fraction {shot_threshold:g}"
            ),
            sum_level=level if math.isfinite(level) else None,
        )

    first_x, first_y, first_sum = xs[:, i0].copy(), ys[:, i0].copy(), sums[:, i0].copy()
    _mask(first_x, first_y, first_sum, lost_at)
    window = (max(i0 - trigger_sample, 0), min(i0 + raw_window_turns, n_samples))
    return ShotExtraction(
        x=first_x, y=first_y, sum=first_sum, lost_at=lost_at, i0=i0, window=window
    )


def average_shots(shots: Sequence[ShotExtraction], *, average: str = "mean") -> AveragedShot:
    """Average the included shots of one shot action.

    Args:
        shots: The included shots, in shot order.
        average: ``"mean"`` or ``"nanmean"``.

    Returns:
        The averaged first turn, masked at the earliest loss.

    Raises:
        ValueError: When there is no shot, the shots differ in BPM count, or
            ``average`` is not one of :data:`AVERAGE_MODES`.
    """
    if average not in AVERAGE_MODES:
        raise ValueError(f"average must be one of {AVERAGE_MODES}; got {average!r}")
    if not shots:
        raise ValueError("there is no included shot to average")
    n_bpm = len(shots[0].x)
    if any(len(shot.x) != n_bpm for shot in shots):
        raise ValueError("the shots differ in BPM count")

    per_shot_lost_at = [shot.lost_at for shot in shots]
    losses = [lost for lost in per_shot_lost_at if lost is not None]
    lost_at = min(losses) if losses else None

    combine = np.mean if average == "mean" else np.nanmean
    with warnings.catch_warnings():
        # nanmean over a BPM that every shot lost warns; that BPM is masked below.
        warnings.simplefilter("ignore", RuntimeWarning)
        x = combine(np.stack([shot.x for shot in shots]), axis=0)
        y = combine(np.stack([shot.y for shot in shots]), axis=0)
        total = combine(np.stack([shot.sum for shot in shots]), axis=0)
    x, y, total = np.array(x, dtype=float), np.array(y, dtype=float), np.array(total, dtype=float)
    _mask(x, y, total, lost_at)

    i0 = int(statistics.median_low([shot.i0 for shot in shots]))
    window = next(shot.window for shot in shots if shot.i0 == i0)
    return AveragedShot(
        x=x,
        y=y,
        sum=total,
        lost_at=lost_at,
        i0=i0,
        window=window,
        per_shot_lost_at=per_shot_lost_at,
        shots_included=len(shots),
    )
