"""The procedural generator's contract: absolute time, VA baselines, bounded.

``osprey.simulation.procedural`` synthesizes history for the channels a
project's ``machine.json`` does not describe — 1,872 of the control-assistant
preset's 2,908 — and its output is written into a real store as well as
returned from live archiver queries. That makes three things load-bearing, and
this file locks each of them:

* **Window independence.** Values are a pure function of ``(channel, epoch
  seconds)``. Two overlapping queries must agree on every timestamp they share,
  and a document seeded at time T must equal what a later query synthesizes for
  T — otherwise materializing the series produces history that contradicts the
  next read. The predecessor generator was a function of the query's *shape*
  (an ``np.linspace(0, 1, n)`` axis) and satisfied neither.

* **Baseline anchoring.** The baseline is whatever the Virtual Accelerator
  boots the channel at, drawn from the same two sources the VA itself uses.
  The per-partition test below reads the shipped channel manifest and machine
  model and checks the generator against the VA's own serving layer, so a
  divergence between the two worlds fails here rather than as a phantom step in
  a deployed archive.

* **A bounded, noise-scaled envelope.** Seeded history and recorded reality
  meet at a seam; the step across it has to be attributable to noise, not read
  as an event. ``deviation_bound`` states the envelope and the shape terms are
  structurally inside it.
"""

import json
import os
import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from osprey.connectors.pv_taxonomy import classify_pv
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels
from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb
from osprey.simulation.procedural import (
    DEFAULT_NOISE_LEVEL,
    KIND_SHAPES,
    baseline_value,
    deviation_bound,
    generate_series,
)
from osprey.simulation.series import epoch_seconds_array

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "src/osprey/services/virtual_accelerator/manifest/channel_manifest.json"
MACHINE = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation/machine.json"

# A fixed instant, so a failure is reproducible rather than time-of-day
# dependent. Any epoch works: the generator has no preferred origin.
T0 = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)

# One PV per kind the taxonomy can return, named the way a real facility names
# them so `classify_pv` reaches the intended branch.
KIND_EXAMPLES = {
    "beam_current": "SR:DCCT:BEAM:CURRENT",
    "current": "SR:MAG:QF01:CURRENT",
    "voltage": "SR:RF:CAV01:VOLTAGE",
    "power": "SR:RF:CAV01:POWER:FWD",
    "pressure": "SR:VAC:IP07:PRESSURE",
    "temperature": "SR:RF:CAV01:TEMP:BODY",
    "lifetime": "SR:BEAM:LIFETIME",
    "position": "SR:DIAG:BPM:12:POSITION:X",
    "energy": "SR:BEAM:ENERGY",
    "default": "SR:SOME:UNCLASSIFIED:CHANNEL",
}


def _sigma(pv: str, noise_level: float = DEFAULT_NOISE_LEVEL) -> float:
    """The channel's noise sigma, the unit every shape amplitude is quoted in.

    Read back out of the public bound rather than recomputed: passing zero
    Gaussian headroom leaves exactly the shape envelope, so a test tolerance
    tracks the shape table instead of restating it.
    """
    return deviation_bound(pv, noise_level=noise_level, noise_sigmas=0.0) / (
        KIND_SHAPES[classify_pv(pv).name].shape_sigmas
    )


def _phase_profile(pv: str, bins: int = 24, cycles: int = 240) -> np.ndarray:
    """The kind's periodic term, recovered by folding many cycles onto one.

    Every sample carries an independent noise draw roughly as large as the
    whole shape envelope — deliberately, since the live machine serves these
    channels as stationary noise and history that visibly exceeded it would be
    a fabricated dynamic. So the shape is not visible sample to sample and a
    test that looked for it there would be testing the noise. Averaging samples
    that share a phase suppresses the noise as ``1/sqrt(n)`` while the periodic
    term survives intact, and spanning many cycles (here ~40 days, far longer
    than any declared drift period) averages the drift stack away with it.

    The result is rotated so its maximum sits first. Each channel's cycle
    starts at a phase drawn from its own key — that is what stops a family of
    channels from peaking in unison — so the phase *origin* is per-channel and
    carries no shape information; only the profile's course away from its peak
    does.

    Returns:
        ``bins`` mean values, ordered by phase from the peak onwards, with the
        overall mean removed so the result is centred on zero.
    """
    period = KIND_SHAPES[classify_pv(pv).name].cycle_period_s
    samples_per_cycle = bins * 10
    grid = _grid(T0, count=cycles * samples_per_cycle, step_s=period / samples_per_cycle)

    values = generate_series(pv, grid)
    phase_bin = np.floor(np.mod(grid / period, 1.0) * bins).astype(int)
    profile = np.array([values[phase_bin == b].mean() for b in range(bins)])
    profile -= profile.mean()
    return np.roll(profile, -int(np.argmax(profile)))


def _grid(start: datetime, count: int, step_s: float) -> np.ndarray:
    """Epoch seconds for ``count`` samples every ``step_s`` from ``start``.

    Built through ``epoch_seconds_array`` from real datetimes, which is the
    conversion every production caller is required to use — computing the
    seconds arithmetically here would test a route the writers do not take.
    """
    stamps = [start + timedelta(seconds=step_s * i) for i in range(count)]
    values = epoch_seconds_array(stamps)
    assert values is not None
    return values


class TestWindowIndependence:
    """The property the whole seeded-store design rests on."""

    def test_overlapping_windows_agree_pointwise(self):
        """Two queries of different span and cadence, sharing timestamps.

        The windows deliberately differ in *both* origin and sample count: a
        generator that keyed on the sample index would pass a same-shape
        comparison and fail this one.
        """
        pv = "SR:VAC:IP07:PRESSURE"
        shared_start = T0 + timedelta(hours=2)

        wide = _grid(T0, count=181, step_s=120.0)
        narrow = _grid(shared_start, count=31, step_s=120.0)

        wide_values = generate_series(pv, wide)
        narrow_values = generate_series(pv, narrow)

        overlap = np.intersect1d(wide, narrow)
        assert len(overlap) == 31, "the two grids were meant to share their tail"

        wide_at_overlap = wide_values[np.searchsorted(wide, overlap)]
        narrow_at_overlap = narrow_values[np.searchsorted(narrow, overlap)]
        assert np.array_equal(wide_at_overlap, narrow_at_overlap)

    def test_a_single_sample_equals_its_value_in_a_batch(self):
        """A recorder reads one timestamp; a seeder writes a whole grid.

        The two must not disagree, so batching may not change a value — which
        also means the scalar (live-read) path and the array path are one
        computation.
        """
        pv = "SR:MAG:QF01:CURRENT"
        grid = _grid(T0, count=97, step_s=45.0)
        batch = generate_series(pv, grid)

        for index in (0, 13, 96):
            assert generate_series(pv, float(grid[index])) == batch[index]

    def test_reordering_the_grid_permutes_the_values(self):
        """No stream position: a shuffled query returns shuffled values."""
        pv = "CRYO:LN2:TEMP"
        grid = _grid(T0, count=64, step_s=30.0)
        order = np.arange(64)[::-1]

        assert np.array_equal(generate_series(pv, grid[order]), generate_series(pv, grid)[order])

    def test_distinct_channels_do_not_collide(self):
        """Reproducible must not mean identical across channels."""
        grid = _grid(T0, count=32, step_s=60.0)
        first = generate_series("SR:VAC:IP07:PRESSURE", grid)
        second = generate_series("SR:VAC:IP08:PRESSURE", grid)

        assert not np.array_equal(first, second)


class TestCrossProcessDeterminism:
    """A seeder in one process and a query in another must agree exactly."""

    def test_values_repeat_in_fresh_interpreters(self):
        """Fresh interpreters with different hash seeds.

        ``hash()`` is salted per process, so only a subprocess can catch a key
        derived from it — the failure mode this construction exists to avoid.
        """
        script = textwrap.dedent("""
            import json
            from datetime import UTC, datetime, timedelta
            from osprey.simulation.procedural import generate_series
            from osprey.simulation.series import epoch_seconds_array

            start = datetime(2026, 3, 14, 9, 26, 53, tzinfo=UTC)
            stamps = [start + timedelta(seconds=17 * i) for i in range(24)]
            grid = epoch_seconds_array(stamps)
            print(json.dumps(generate_series("SR:VAC:IP07:PRESSURE", grid).tolist()))
        """)

        runs = []
        for seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            runs.append(json.loads(result.stdout.strip().splitlines()[-1]))

        assert runs[0] == runs[1] == runs[2]
        assert len(runs[0]) == 24
        # The in-process value is the same computation, so it must match too:
        # cross-process agreement among three subprocesses would be vacuous if
        # the host process disagreed with all of them.
        grid = _grid(T0, count=24, step_s=17.0)
        assert generate_series("SR:VAC:IP07:PRESSURE", grid).tolist() == runs[0]


class TestBaselineAnchoring:
    """Baselines come from the same two sources the VA boots channels from.

    The manifest's three fidelity partitions are covered together because they
    differ precisely in where their boot value comes from: pyat-coupled and
    sp-echo channels are ``:SP``/``:RB`` addresses seeded from the machine
    model, while static-noisy channels have no model entry and are served from
    the PV taxonomy. A generator that anchored only one of them would seam
    cleanly on that partition and plant a step on the others.
    """

    @staticmethod
    def _manifest_channels() -> list[dict]:
        return json.loads(MANIFEST.read_text())["channels"]

    @staticmethod
    def _boot_values() -> dict[str, float]:
        """The map the VA entrypoint derives for ``build_serving_pvdb``."""
        return {
            address: entry["value"]
            for address, entry in load_machine_json_channels(MACHINE).items()
            if "value" in entry
        }

    @staticmethod
    def _va_value(channel: dict, boot_values: dict, served: dict) -> float:
        """What the VA settles this channel at, by its own three rules.

        Written as the VA's rules rather than as the generator's, so agreement
        is evidence and not a tautology. ``served`` is the record database the
        VA builds at boot; it is the authority for the undriven channels, whose
        boot value is also their steady value. The static-noisy partition is
        *not* read from it: those records are overwritten by ``EngineSource``
        within one poll tick, so their boot spec is a value nobody observes.
        """
        address = channel["address"]
        if address in boot_values:
            return float(boot_values[address])
        if channel["subfield"] in ("SP", "RB"):
            return float(served[address]["value"])
        return classify_pv(address).base_value

    @pytest.mark.parametrize("partition", ["pyat-coupled", "sp-echo", "static-noisy"])
    def test_baseline_equals_the_value_the_va_settles_the_channel_at(self, partition):
        """Every analog channel of the partition, not a sample of them.

        The manifest is the deployed channel universe and it is cheap to sweep
        whole; a sampled version of this test would pass while leaving a
        fabricated baseline on any channel it skipped.
        """
        channels = [c for c in self._manifest_channels() if c["partition"] == partition]
        assert channels, f"no {partition} channels in the shipped manifest"

        boot_values = self._boot_values()
        served = build_serving_pvdb(self._manifest_channels(), boot_values=boot_values).pvdb

        analog = [c for c in channels if c["record_type"] == "ai"]
        assert analog, f"no analog {partition} channels in the shipped manifest"

        for channel in analog:
            address = channel["address"]
            assert baseline_value(address, boot_values) == pytest.approx(
                self._va_value(channel, boot_values, served)
            ), f"{address} ({partition}) starts somewhere the VA does not"

    def test_an_unseeded_setpoint_is_not_given_a_taxonomy_guess(self):
        """The rule the sweep above would let through if it were sampled.

        A dozen sp-echo setpoints have no machine-model seed. Nothing drives a
        setpoint but a client write, so the live channel reads 0 forever —
        history at the taxonomy's 5 kV guess for a name containing "VOLTAGE"
        would be invention, and exactly the kind an agent would diagnose from.
        """
        address = "SR:VAC:ION-PUMP:01:VOLTAGE:SP"
        boot_values = self._boot_values()
        assert address not in boot_values
        assert classify_pv(address).base_value == 5000.0

        assert baseline_value(address, boot_values) == 0.0
        assert np.array_equal(
            generate_series(address, _grid(T0, count=16, step_s=60.0), baseline=0.0),
            np.zeros(16),
        )

    def test_a_seeded_setpoint_overrides_the_taxonomy_guess(self):
        """The override has to actually bite, or the test above proves nothing.

        A magnet ``:SP`` classifies as the ``current`` kind (base 150 A) while
        the machine model seeds it at its real value; the seeded one wins.
        """
        boot_values = self._boot_values()
        address = "SR:MAG:HCM:01:CURRENT:SP"
        assert address in boot_values, "the shipped machine model no longer seeds this"
        assert boot_values[address] != classify_pv(address).base_value

        assert baseline_value(address, boot_values) == pytest.approx(boot_values[address])
        assert baseline_value(address) == classify_pv(address).base_value

    def test_an_unseeded_channel_falls_back_to_the_taxonomy(self):
        """What the VA's own ``EngineSource`` serves for a modelless channel."""
        pv = "BR:DIAG:BPM:01:POSITION:X"
        assert baseline_value(pv, {}) == classify_pv(pv).base_value
        assert baseline_value(pv, None) == classify_pv(pv).base_value

    def test_a_non_numeric_seed_is_ignored_rather_than_coerced(self):
        """String-valued channels exist in machine models; this one is numeric."""
        pv = "SR:MAG:QF01:CURRENT"
        assert baseline_value(pv, {pv: "not a number"}) == classify_pv(pv).base_value

    def test_the_series_is_centred_on_the_anchored_baseline(self):
        """Anchoring the baseline is pointless if the series drifts off it.

        The tolerance is the shape envelope (``noise_sigmas=0``) plus the
        standard error of the mean, which is far tighter than the seam bound —
        a series that sat a whole sigma off its anchor would pass that and fail
        this.
        """
        pv = "SR:MAG:HCM:01:CURRENT:SP"
        anchored = 0.734
        count = 2048
        grid = _grid(T0, count=count, step_s=30.0)

        values = generate_series(pv, grid, baseline=anchored)

        tolerance = deviation_bound(pv, baseline=anchored, noise_sigmas=4.0 / np.sqrt(count))
        assert values.mean() == pytest.approx(anchored, abs=tolerance)


class TestSeamEnvelope:
    """The excursion stays inside the band a seam assertion is allowed to quote."""

    @pytest.mark.parametrize("pv", sorted(KIND_EXAMPLES.values()))
    def test_every_sample_is_inside_the_declared_bound(self, pv):
        """Over a span far longer than any shape period, so no term is missed."""
        grid = _grid(T0, count=4096, step_s=120.0)  # ~5.7 days

        values = generate_series(pv, grid)
        base = baseline_value(pv)

        assert np.all(np.isfinite(values))
        assert np.max(np.abs(values - base)) <= deviation_bound(pv)

    def test_the_bound_scales_with_the_declared_noise(self):
        """The envelope is a multiple of the channel's sigma, not a constant."""
        pv = "SR:RF:CAV01:TEMP:BODY"
        assert deviation_bound(pv, noise_level=0.02) == pytest.approx(
            2.0 * deviation_bound(pv, noise_level=0.01)
        )

    def test_zero_noise_level_yields_the_bare_baseline(self):
        """A noise level of zero is an explicit request for determinism.

        The VA serves the bare boot value there, so its history must be the
        same flat line — a textured history under a channel the live machine
        holds perfectly still is exactly the fabricated motion this generator
        exists to stop producing.
        """
        pv = "SR:RF:CAV01:VOLTAGE"
        grid = _grid(T0, count=128, step_s=60.0)

        values = generate_series(pv, grid, noise_level=0.0)

        assert np.array_equal(values, np.full(128, baseline_value(pv)))
        assert deviation_bound(pv, noise_level=0.0) == 0.0

    def test_a_zero_baseline_channel_still_moves(self):
        """A BPM's baseline is 0.0, which kills a purely relative sigma.

        The kind's absolute noise floor is what keeps it alive; without it the
        channel reads dead flat however noisy it is declared to be.
        """
        pv = "SR:DIAG:BPM:12:POSITION:X"
        assert baseline_value(pv) == 0.0

        values = generate_series(pv, _grid(T0, count=256, step_s=10.0))

        assert values.std() > 0.0
        assert np.max(np.abs(values)) <= deviation_bound(pv)


class TestKindShapes:
    """Each kind keeps a recognizable, individual signature."""

    def test_every_taxonomy_kind_has_a_shape(self):
        """A kind added to the taxonomy without one here would fall back
        silently to the generic texture; the example map is the checklist."""
        for kind_name, pv in KIND_EXAMPLES.items():
            assert classify_pv(pv).name == kind_name, f"{pv} no longer classifies as {kind_name}"
            assert kind_name in KIND_SHAPES

    @pytest.mark.parametrize(("kind_name", "pv"), sorted(KIND_EXAMPLES.items()))
    def test_each_kind_varies_around_its_baseline(self, kind_name, pv):
        """Finite, moving, and centred — the sanity floor for every branch.

        The sigma the series moves by is the kind's declared one, not a scale
        this generator invented: a kind whose shape amplitudes were mistyped in
        engineering units instead of sigmas would sail past a "varies" check
        and fail the standard-deviation bound here.
        """
        count = 1024
        grid = _grid(T0, count=count, step_s=60.0)
        base = baseline_value(pv)
        sigma = _sigma(pv)

        values = generate_series(pv, grid)

        assert np.all(np.isfinite(values))
        # At least the noise term's own sigma, and no more than a small
        # multiple of it once the bounded shape terms are added on top.
        assert sigma <= values.std() <= 3.0 * sigma
        assert values.mean() == pytest.approx(
            base, abs=deviation_bound(pv, noise_sigmas=4.0 / np.sqrt(count))
        )

    def test_beam_current_sawtooths_between_refills(self):
        """The kind's signature: a slow decay broken by a sudden recovery.

        The profile is recovered by folding many cycles onto one phase axis
        (see :func:`_phase_profile`). A refill cycle is a *ramp* on that axis:
        it descends from the refill all the way to the trough immediately
        before the next one, which is what tells it apart from any other
        periodic wiggle of the same amplitude.
        """
        pv = KIND_EXAMPLES["beam_current"]
        profile = _phase_profile(pv)
        sigma = _sigma(pv)

        # The channel's phase offset is keyed, so the refill lands inside a bin
        # rather than on a boundary: the last bin of the peak-aligned profile
        # is the one straddling the jump and averages both sides of it. The
        # decay is the other bins.
        decay, straddling = profile[:-1], profile[-1]

        assert np.all(np.diff(decay) < 0), "the cycle is not a monotone decay"
        # The declared span is 2 x cycle_sigmas = 3 sigma, less the bin widths
        # the fold averages across; the drift stack leaves a small residual.
        assert decay[0] - decay[-1] > 2.0 * sigma
        assert decay[-1] < straddling < decay[0], "the jump is not inside that bin"

    def test_voltage_swings_smoothly_rather_than_sawtoothing(self):
        """The contrasting shape, on the same phase axis.

        A sine-shaped kind returns to where it started after one period, so its
        folded profile has no jump: the two ends nearly meet, and the extremes
        sit about half a period apart. Sawtooth and sine are the only two
        shapes the table can declare, so this pins the other branch — without
        it, a table that declared every kind a sawtooth would still pass.
        """
        pv = KIND_EXAMPLES["voltage"]
        profile = _phase_profile(pv)
        sigma = _sigma(pv)
        bins = len(profile)

        assert abs(profile[0] - profile[-1]) < 0.5 * sigma
        separation = abs(int(np.argmax(profile)) - int(np.argmin(profile)))
        assert abs(separation - bins // 2) <= 2, "extremes are not half a period apart"

    def test_the_default_kind_catches_unclassified_channels(self):
        """Most of the served namespace is status and reference channels."""
        pv = KIND_EXAMPLES["default"]
        assert classify_pv(pv).name == "default"

        values = generate_series(pv, _grid(T0, count=64, step_s=60.0))

        assert np.all(np.isfinite(values))
        assert values.std() > 0.0


def test_default_noise_level_matches_what_the_va_serves():
    """The two halves of a deployed world must assume the same noise.

    The constant is duplicated rather than imported (``osprey.simulation`` must
    load without the VA service package), so nothing but this assertion keeps
    the two from drifting apart — and a drift would show up as a seam step
    between seeded history and recorded reality.
    """
    from osprey.services.virtual_accelerator.ioc.engine_source import DEFAULT_NOISE_LEVEL as VA

    assert DEFAULT_NOISE_LEVEL == VA
